TARGET_PDF=report.pdf
MAIN_TEX=main.tex
OUT_DIR=out
MAIN_PDF=$(OUT_DIR)/main.pdf
DEPS_FILE=$(OUT_DIR)/deps.mk


# target: help
.PHONY: help
help:
	@echo "available targets:"
	@echo ""
	@echo "- report     build the report.pdf"
	@echo "- plots      generate the plots in media/"
	@echo "- clean      clean all temporary LaTeX outputs and the report.pdf"


# target: report
.PHONY: report
report: $(TARGET_PDF)

$(TARGET_PDF): $(MAIN_PDF)
	cp $(MAIN_PDF) $@

$(MAIN_PDF):
	mkdir -p $(OUT_DIR)/content
	find . -type d ! -path './$(OUT_DIR)*' -exec mkdir -p $(OUT_DIR)/\{\} \;
	latexmk -lualatex \
	 	-outdir=$(OUT_DIR) \
		-deps-out=$(DEPS_FILE) \
		-shell-escape \
		-synctex=1 \
		-interaction=nonstopmode \
		-file-line-error \
		$(MAIN_TEX)
	touch $(MAIN_PDF)

-include $(DEPS_FILE)


# target: plots
plots:
	sh media/generate_plots.sh


# target: clean
clean:
	rm -rf $(TARGET_PDF) $(OUT_DIR)/
