# JSS Information for Authors

Source: https://www.jstatsoft.org/authors

## Requirements
Our [mission page](https://www.jstatsoft.org/pages/view/mission#types-of-papers) states what type of papers are suitable for publication in this journal. Articles describing a novel software contribution must cover implementations with a similar scope (including those in other languages and for other statistical software environments), and discuss the advantages and disadvantages of the new contribution compared to existing implementations, when possible and also applicable using empirical illustrations. Extensive simulation studies are discouraged.
All new submissions must provide the following attachments:
1. PDF manuscript in JSS style,
2. source code for the software and
3. replication materials for all results from the manuscript,
preferably via a single, commented standalone replication script.
The replication materials must enable reproducibility of all results from the manuscript (see also [Review Process](https://www.jstatsoft.org/authors#review-process)). In case the attachments are too large for upload, this should be explained in the submission and a link should be provided.

## Software Preparation
Source code must be submitted in ASCII files. It should be readable and have comments. For software environments with package/library systems (e.g., R, Python, Stata, etc.), the code needs to come with formatted help files and be packaged for easy installation. If at all possible, submitted software should also be available from a standard repository like CRAN, Bioconductor or PyPI (and not just GitHub, GitLab, etc). For R packages, we encourage inclusion of JSS submissions as vignettes in the package. For R packages with functions that create compound objects, a minimum expectation is that R's classes and methods systems are leveraged, e.g., by making use of S3 classes and providing standard methods (such as print, plot, and summary). Software authors may also find it helpful to follow the advice in the guidelines described in the [rOpenSci Statistical Software Peer Review](https://stats-devguide.ropensci.org/) book. It is not necessary for a JSS submission to fully conform with all the standards described there, though.
Please note that JSS only allows file uploads up to a maximum size of 50 megabyte (MB). If there is a need for bigger supplementary files, the authors should include a file containing the download instructions with a link to the external data source. If the authors do not have the resources to provide these files on an external server, please contact the JSS editors at [editor@jstatsoft.org](mailto:editor@jstatsoft.org).
Code can be in any language. The majority of software published in JSS is written in S (R), MATLAB, SAS/IML, C++, Java, or Python. It will be more difficult to find appropriate referees for software written in less popular languages and authors might have to accept longer review times.
Code needs to include the GNU General Public Licence (GPL), versions GPL-2 or GPL-3, or a GPL-compatible license for publication in JSS.

## Manuscript Preparation
Manuscripts must be written in English using the LaTeX text processing system. The [[JSS Style Guide](https://www.jstatsoft.org/style)](https://www.jstatsoft.org/style) must be followed for formatting manuscripts. Only PDF files can be submitted. It is the responsibility of the authors to provide a submission in the appropriate format, paying very close attention to the style instructions. Manuscripts not meeting the formal requirements will be returned to the authors immediately. JSS has no resources to help authors with conversions from other typesetting environments, so we strongly advise authors who need assistance to find a local LaTeX expert. We provide templates, detailed instructions and a FAQ in our JSS Style Guide. There is no page limit, nor a limit on the number of figures or tables. Manuscripts longer than 30 pages typically require much longer review times. However, more important for the review times incurred might be that the manuscript is carefully written exhibiting high readability.
All figures, tables, and other output presented in the manuscript must be fully and exactly reproducible on at least one platform. Please indicate any platform dependencies with the submission. Make sure to initialize the generator for pseudo random numbers if the results rely on some form of simulation. Typically, reproducibility is demonstrated by a standalone replication script. As a rough guideline, anybody with sufficient technical background should be able to install the software and to run the replication material within reasonable time (one hour on a regular PC). In case there are good reasons that replication should take considerably longer or requires special hardware, the authors should in addition to the “full” replication script supply a script that replicates similar results within reasonable time on a regular PC. To facilitate review, authors are strongly encouraged to provide an output file that shows the results from running the single standalone replication script so that this can be compared against the results presented in the manuscript. For R submissions, this should be done by providing a file “code.html” created by running knitr::spin(“code.R”) on the replication script “code.R” which should include a call to sessionInfo() at the end.

## Online Submission
Contributions are submitted online. Authors should have a look at our [step-by-step submission guide](https://www.jstatsoft.org/guides/submission).

## Review Process
The software and manuscript will be peer reviewed by the statistical software community. The editor-in-chief selects a section editor, the section editor selects two reviewers (one of whom can be the section editor). The review has two parts: both the software and the manuscript are reviewed. The software should work as indicated, be clearly documented, and serve a useful purpose. Reviewers are instructed to evaluate both correctness and usefulness. Reviewers are expected to verify that manuscripts are reproducible, and to only recommend conditional acceptance after reproducibility problems have been completely resolved.
Reviewers of R packages may find it helpful to follow the guidelines described in the [rOpenSci Statistical Software Peer Review](https://stats-devguide.ropensci.org/) book. The standards described there provide many useful guidelines; however, they do not form a requirement for packages reviewed for this journal.
The possible editorial decisions are:
Reject Decisions
- decline (editorial reject),
- reject (after review, possibly after revision)
Revise Decisions
- resubmit for review (will go back to both reviewers and
editor),
- revision required (will only go back to editor)
Accept Decisions
- (conditional) accept
Revisions, in any state of the review process, must always be submitted together with a point-to-point reply to the points made by reviewers and editors.
Manuscripts rejected by JSS cannot be resubmitted. We hope the review process gives the authors sufficient information to submit to another journal.
If a revision is not received back from the authors within six months of a revision being requested, the paper will be considered to be withdrawn.
If a conditionally accepted paper is not brought into final form within a year, the paper will be considered to be withdrawn.

## Publishing Process
Accept decisions are always conditional on delivering a manuscript that satisfies the JSS style and reproducibility requirements. It is the responsibility of the author to make sure the manuscript conforms with the JSS requirements. Accepted manuscripts that do not satisfy these requirements cannot be published. Each published article will be a separate issue. Issues are published in volumes.
