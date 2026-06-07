# 編譯整合報告。用法：make（產生 final_report.pdf）、make clean。
TEX := final_report

.DEFAULT_GOAL := pdf

.PHONY: pdf
pdf:  ## xelatex 編譯（跑兩次處理 toc）
	xelatex -interaction=nonstopmode -halt-on-error $(TEX).tex
	xelatex -interaction=nonstopmode -halt-on-error $(TEX).tex

.PHONY: clean
clean:  ## 清除編譯暫存（保留 pdf）
	rm -f $(TEX).aux $(TEX).log $(TEX).toc $(TEX).out

.PHONY: distclean
distclean: clean  ## 連 pdf 一起清除
	rm -f $(TEX).pdf
