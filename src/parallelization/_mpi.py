from __future__ import annotations
import numpy as np
import mpi4py.MPI as mpi
import src.boundary as bdry


class MpiDomain2D:
    """
    Helper class providing methods and properties that ease the decomposition of a 2D domain in order to parallelize it
    using MPI.
    """

    def __init__(self, lattice_size: tuple[int, int], decomposition: tuple[int, int]):
        """
        Parameters
        ----------
        lattice_size : tuple[int, int]
            A tuple ``(X, Y)`` describing the lattice dimensions.

        decomposition : tuple[int, int]
            A tuple ``(N, M)`` describing the domain composition of the lattice.
            This means the X×Y lattice gets decomposed into a N×M grid where each subdomain gets handled by a separate
            process.
        """
        self._global_x, self._global_y = lattice_size
        self._decomp_x, self._decomp_y = self._determine_decomposition(lattice_size, decomposition)

        # create a cartesian communicator
        self._comm = mpi.COMM_WORLD.Create_cart(dims=(self._decomp_y, self._decomp_x),
                                                periods=(False, False),
                                                reorder=True)
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

        # get the properties of the subdomain/process w.r.t. the domain decomposition
        self._coord_y, self._coord_x = self._comm.Get_coords(self._rank)
        self._size_x = self._compute_size_1d(self._global_x, self._decomp_x, self._coord_x)
        self._size_y = self._compute_size_1d(self._global_y, self._decomp_y, self._coord_y)
        self._padding = np.array([self._compute_padding_1d(self._coord_y, self._decomp_y),
                                  self._compute_padding_1d(self._coord_x, self._decomp_x)])
        self._physical_domain = (slice(self._padding[0, 0], -self._padding[0, 1] or None),
                                 slice(self._padding[1, 0], -self._padding[1, 1] or None))
        self._buffered_subdomain_size = (self._size_y + np.sum(self._padding[0]),
                                         self._size_x + np.sum(self._padding[1]))
        self._range_x = self._compute_range_1d(self._global_x, self._decomp_x, self._coord_x)
        self._range_y = self._compute_range_1d(self._global_y, self._decomp_y, self._coord_y)

        # get ranks of neighboring subdomains for communication
        self._src_left, self._dest_left = self._comm.Shift(1, -1)
        self._src_right, self._dest_right = self._comm.Shift(1, 1)
        self._src_down, self._dest_down = self._comm.Shift(0, 1)
        self._src_up, self._dest_up = self._comm.Shift(0, -1)

    @property
    def comm(self) -> mpi.Cartcomm:
        """
        Reference to the MPI communicator instance.
        """
        return self._comm

    @property
    def rank(self) -> int:
        """
        The process rank in range ``[0; n)`` where ``n`` is the number of MPI processes.
        """
        return self._rank

    @property
    def size(self) -> int:
        """
        The number of MPI processes.
        """
        return self._size

    @property
    def decomposition(self) -> tuple[int, int]:
        """
        The domain decomposition in ``(X, Y)`` direction.
        """
        return self._decomp_x, self._decomp_y

    @property
    def padding(self) -> np.ndarray:
        """
        A 2x2 matrix definition the padding around the subdomain:

        - ``padding[0]`` gives the padding in y-direction
        - ``padding[1]`` gives the padding in x-direction

        The value can be passed directly to :method:`numpy.pad`.
        """
        return self._padding

    @property
    def physical_domain(self) -> tuple[slice, slice]:
        """
        A tuple of two slices that can be used to extract the physical simulation domain without the padding/ghost cells
        around it, that are only used for communication between the MPI processes.
        """
        return self._physical_domain

    @property
    def buffered_subdomain_size(self) -> tuple[int, int]:
        """
        The size of the subdomain including the buffer rows and columns for the communication.

        Returns
        -------
        tuple[int, int]
            A tuple ``(size Y, size X)``.
        """
        return self._buffered_subdomain_size

    @property
    def subdomain_selection(self) -> tuple[slice, slice]:
        """
        A tuple of 2 slices defining the selection of the physical part of the subdomain from the global lattice domain
        in y and x direction respectively.
        """
        return (slice(self._range_y[0], self._range_y[1]),
                slice(self._range_x[0], self._range_x[1]))

    def _determine_decomposition(self, lattice_size: tuple[int, int], decomposition: tuple[int, int]) -> tuple[int, int]:
        """
        Determine the decomposition of the lattice domain in a flexible manner.

        Parameters
        ----------
        lattice_size : tuple[int, int]
            The size of the lattice, i.e. the global domain size, as ``(X, Y)`` tuple.

        decomposition : tuple[int, int]
            The user-specified decomposition of the domain as ``(x, y)`` tuple.

        Returns
        -------
        tuple[int, int]
            - if ``(x>0, y>0)``, then ``decomposition``
            - if ``(x>0, y<0)``, then ``(x, y//size)``
            - if ``(x<0, y>0)``, then ``(x//size, y)``
            - if ``(x<0, y<0)``, then the optimal decomposition such that ``x/y`` approximates ``X/Y`` best
        """
        gx, gy = lattice_size
        dx, dy = decomposition
        size = mpi.COMM_WORLD.size
        if dx < 0 and dy < 0:
            # auto-determine the best decomposition based on the optimal domain ratio (in (0;1])
            ratio = gx/gy if gx <= gy else gy/gx
            # initialize smaller and larger side decomposition
            sm, lg = 1, size
            for i in range(2, size//2 + 1):
                # skip non-valid combinations
                if size % i != 0:
                    continue
                # stop once the sm/lg ratio has exceeded the optimal ratio
                if i / float(size//i) > ratio:
                    break
                sm, lg = i, size // i
            dx, dy = (sm, lg) if gx <= gy else (lg, sm)
        else:
            # check for sane values
            if (dx < 0 and size % dx != 0) or (dy < 0 and size % dy != 0) or (dx * dy != size):
                raise ValueError(f"MPI: size={size} and decomposition (x={dx}, y={dy}) does not work")
            # check if one value is implicit
            if dx < 0:
                dy = size // dx
            elif dy < 0:
                dx = size // dy
        return dx, dy

    def _compute_size_1d(self, global_size: int, decomposition_size: int, coordinate: int) -> int:
        """
        Compute the size of the subdomain handled by the current process in one dimension.
        The considered dimension is given by the choice of parameters.

        Parameters
        ----------
        global_size : int
            Size of the lattice.

        decomposition_size : int
            Size of the decomposition grid.

        coordinate : int
            Coordinate of the current rank. Must be in the range [0; ``decomposition_size``).

        Returns
        -------
        int
            The size of the subdomain.
        """
        if not 0 <= coordinate < decomposition_size:
            raise ValueError(f"invalud argument: coordinate={coordinate} must be inside the range " +
                             f"[0; decomposition_size={decomposition_size})")
        # if the lattice size (global_size) is not divisible by the number of domains (decomposition_size), the first
        # (global_size % decomposition_size) subdomains are 1 node larger in order to cover the whole lattice
        if coordinate < global_size % decomposition_size:
            return global_size // decomposition_size + 1
        return global_size // decomposition_size

    def _compute_range_1d(self, global_size: int, decomposition_size: int, coordinate: int) -> tuple[int, int]:
        """
        Compute the index range of the subdomain handled by the current process in one dimension.
        The considered dimension is given by the choice of parameters.

        Parameters
        ----------
        global_size : int
            Size of the lattice.

        decomposition_size : int
            Size of the decomposition grid.

        coordinate : int
            Coordinate of the current rank. Must be in the range [0; ``decomposition_size``).

        Returns
        -------
        tuple[int, int]
            A tuple with the start and end of the range as first and second values respectively.
            The end value is excluding.
        """
        lower = sum([self._compute_size_1d(global_size, decomposition_size, i) for i in range(coordinate)])
        upper = lower + self._compute_size_1d(global_size, decomposition_size, coordinate)
        return lower, upper

    def _compute_padding_1d(self, coordinate: int, decomposition_size: int) -> tuple[int, int]:
        """
        Determine whether there is padding required around a subdomain.

        Parameters
        ----------
        coordinate : int
            Coordinate of the current rank. Must be in the range [0; ``decomposition_size``).

        decomposition_size : int
            Size of the decomposition grid.

        Returns
        -------
        tuple[int, int]
            A tuple ``(before, after)`` definition the padding before and after the subdomain.
        """
        return (1 if coordinate > 0 else 0,
                1 if coordinate < decomposition_size-1 else 0)

    def filter_boundaries(self, boundaries: bdry.Boundary) -> bdry.Boundary:
        """
        Filter a boundary flag for the global lattice to contain only the boundaries the current subdomain is adjacent
        to.

        Parameters
        ----------
        boundaries : bdry.Boundary
            The :class:`Boundary` enum flag that should be filtered.

        Returns
        -------
        Boundary
            Filtered :class:`Boundary` enum flag.
        """
        b = ""
        if boundaries & bdry.Boundary.LEFT and self._coord_x == 0:
            b += "l"
        if boundaries & bdry.Boundary.RIGHT and self._coord_x == self._decomp_x-1:
            b += "r"
        if boundaries & bdry.Boundary.TOP and self._coord_y == 0:
            b += "t"
        if boundaries & bdry.Boundary.BOTTOM and self._coord_y == self._decomp_y-1:
            b += "b"
        return bdry.Boundary(b)

    def save_mpiio_2d(self, filename: str, a_ij):
        """
        Write a global two-dimensional array to a single file in the `npy format
        <https://numpy.org/devdocs/reference/generated/numpy.lib.format.html>`_ using MPI I/O.

        Arrays written with this function can be read with ``numpy.load``.

        Parameters
        ----------
        filename : str
            File name.

        a_ij : array_like
            Portion of the array on this MPI processes. This needs to be a 2D array.
        """
        # see: https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#format-version-1-0
        # summary:
        # - first 6+2 bytes: magic string + file format version
        # - next 2 bytes: header length HEADER_LEN
        # - next HEADER_LEN bytes: header dictionary
        #   - to store dtype, order and shape
        #   - padded with spaces and terminated with \n to make
        #       len(magic string) + 2 + len(length) + HEADER_LEN
        #     divisible by 64 for alignment purposes
        # - following the header comes the array data

        # first 6+2 bytes are the magic strings + file format version
        magic_str = np.lib.format.magic(1, 0)

        # get global domain size
        local_nx, local_ny = a_ij.shape
        nx = np.empty_like(local_nx)
        ny = np.empty_like(local_ny)
        commx = self._comm.Sub((True, False))
        commy = self._comm.Sub((False, True))
        commx.Allreduce(np.asarray(local_nx), nx)
        commy.Allreduce(np.asarray(local_ny), ny)

        # header
        arr_dict_str = str({
            "descr": np.lib.format.dtype_to_descr(a_ij.dtype),
            "fortran_order": False,
            "shape": (nx.item(), nx.item())
        })
        # +2 correspondts to len(header) for npy version 1.0
        # the +2 for the version bytes is already included in len(magic_str)
        while (len(magic_str) + 2 + len(arr_dict_str)) % 64 != 63:
            arr_dict_str += " "
        arr_dict_str += "\n"
        header_len = len(magic_str) + 2 + len(arr_dict_str)

        # calculate offset inside file
        offsetx = np.zeros_like(local_nx)
        commx.Exscan(np.asarray(ny*local_nx), offsetx)
        offsety = np.zeros_like(local_ny)
        commy.Exscan(np.asarray(local_ny), offsety)

        # white npy file
        file = mpi.File.Open(self._comm, filename, mpi.MODE_CREATE | mpi.MODE_WRONLY)
        if self._rank == 0:
            file.Write(magic_str)
            file.Write(np.int16(len(arr_dict_str)))
            file.Write(arr_dict_str.encode("latin-1"))
        mpitype = mpi._typedict[a_ij.dtype.char]
        filetype = mpitype.Create_vector(local_nx, local_ny, ny)
        filetype.Commit()
        file.Set_view(header_len + (offsety+offsetx)*mpitype.Get_size(), filetype=filetype)
        file.Write_all(a_ij.copy())
        filetype.Free()
        file.Close()

    def get_grid(self, rank: int = 0) -> np.ndarray:
        """
        Returns
        -------
        numpy.ndarray
            A 2D array descibing the rank arangement. Axis 0 and 1 represent x and y respectively.
        """
        
        # get coordinates and neighbors of current process
        coords = np.array(self._comm.Get_coords(self._rank))
        neighbors = np.array([
            self._comm.Shift(0, 1)[1],
            self._comm.Shift(0, -1)[1],
            self._comm.Shift(1, 1)[1],
            self._comm.Shift(1, -1)[1],
        ])

        # gather coordinates and neighbors of all processes
        all_coords = np.zeros((self._size, 2), dtype=int)
        self._comm.Gather(sendbuf=coords, recvbuf=all_coords, root=0)
        all_neighbors = np.zeros((self._size, 4), dtype=int)
        self._comm.Gather(sendbuf=neighbors, recvbuf=all_neighbors, root=0)

        if self._rank == rank:
            # fill grid according to the coordinates and neighbors
            grid = np.ones((self._decomp_x, self._decomp_y), dtype=int)*-1
            for r in range(self._size):
                y, x = all_coords[r]
                # n - neighbor, x/y - axis, p/m - positive/negative shift
                nyp, nym, nxp, nxm = all_neighbors[r]
                # for all 4 neighbors do:
                # 1. assert that multiple neighbor relations must give the same result (sanity check)
                # 2. fill grid with neighbor rank
                if nxp >= 0:
                    assert grid[x+1, y] in [nxp, -1]
                    grid[x+1, y] = nxp
                if nxm >= 0:
                    assert grid[x-1, y] in [nxm, -1]
                    grid[x-1, y] = nxm
                if nyp >= 0:
                    assert grid[x, y+1] in [nyp, -1]
                    grid[x, y+1] = nyp
                if nym >= 0:
                    assert grid[x, y-1] in [nym, -1]
                    grid[x, y-1] = nym

            return grid
