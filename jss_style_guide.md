# JSS Style Guide

Source: https://www.jstatsoft.org/style

## LaTeX
When using plain LaTeX, you may start from the following article template: [jss-article-tex.zip](https://www.jstatsoft.org/public/journals/1/jss-article-tex.zip) (including all necessary style files). The source file article.tex can be edited in any text editor or dedicated LaTeX editor (including RStudio). To produce the manuscript pdfLaTeX should be used.
The corresponding references are in the BibTeX bibliography ref.bib. The replication code article.R has to be prepared separately and should ideally include comments that make it easy to match it to the manuscript.

## Code Snippets
Code snippet contributions are produced in the same way as articles. You only have to switch the from `\documentclass[article]{jss}` to `\documentclass[codesnippet]{jss}` in the first line of the LaTeX document.

## R/LaTeX via Sweave or knitr
For producing a JSS article with R/Latex, please start from the following template: [jss-article-rnw.zip](https://www.jstatsoft.org/public/journals/1/jss-article-rnw.zip).
If knitr is used to prepare the article.Rnw file, the `render_sweave()` hook should be used to ensure JSS style code formatting.

## Important Style Guidelines
- The manuscript can be compiled by pdfLaTeX.
- `\proglang`, `\pkg` and `\code` have been used for highlighting throughout the paper (including titles and references), except where explicitly escaped.
- References are provided in a .bib BibTeX database and included in the text by `\cite`, `\citep`, `\citet`, etc.
- Titles and headers are formatted as described in the JSS manual:
    - `\title` in title style,
    - all titles in the BibTeX file in title style.
    - `\section`, `\subsection`, etc. in sentence style,
    - annotations of figures/tables (including captions) in sentence style
- Figures, tables and equations are marked with a `\label` and referred to by `\ref`, e.g., `Figure~\ref{...}`.
- Software packages are `\cite{}`d properly.

## Capitalization Rules
- **Sentence style**: Only the first word in a title is capitalized, as is the first word after a colon or a hyphen. Proper names remain in upper case. Used for section headers, captions.
- **Title style**: All principal words should be capitalized. Used for the main title and BibTeX titles.

## Citing Software
If there is no recommended citation, please cite the corresponding manual or webpage.
Example:
```latex
@Manual{SAS-STAT,
  author = {{\proglang{SAS} Institute Inc.}},
  title = {\proglang{SAS/STAT} Software, Version~9.1},
  year = {2003},
  address = {Cary, NC},
  url = {https://www.sas.com/}
}
```

## Citing R Packages
Check `citation("foo")` or CRAN. Use `\proglang`, `\pkg`, `\code`.

## Code Formatting
- Code should preferably be presented in the usual text flow.
- Include spaces before and after operators and after commas.
- `y = a + b * x` (not `y=a+b*x`)
- Code presented in the manuscript should not contain comments within the verbatim code. Comments should be in the normal LaTeX text.
- `\code{...}` for inline code.
- `{Code}`, `{CodeInput}`, `{CodeOutput}` environments for blocks.
- Code must fit within textwidth.
