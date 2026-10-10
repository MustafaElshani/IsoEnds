#!/usr/bin/env Rscript
# limma moderated t-test, one value per unit of replication.
#
# Usage: Rscript limma_moderated_t.R values.tsv out.tsv ROBUST [GROUPS]
#   values.tsv  features x units, first column "feature"; NA for missing
#   ROBUST      TRUE or FALSE (eBayes robust = ...)
#   GROUPS      optional, one 0 (control) or 1 (treated) per unit, comma-separated: tests treated - control;
#               without it, tests each feature's mean against zero
suppressPackageStartupMessages(library(limma))
args <- commandArgs(trailingOnly = TRUE)
X <- as.matrix(read.delim(args[1], row.names = 1, check.names = FALSE))
robust <- toupper(args[3]) == "TRUE"
if (length(args) >= 4) {
  design <- cbind(control = 1, treated = as.numeric(strsplit(args[4], ",")[[1]]))
  coef <- 2
} else {
  design <- matrix(1, ncol(X), 1, dimnames = list(colnames(X), "mean"))
  coef <- 1
}
fit <- eBayes(lmFit(X, design), robust = robust)
res <- topTable(fit, coef = coef, number = Inf, sort.by = "none")
res$n_units <- rowSums(!is.na(X))
res$df_residual <- fit$df.residual
res$df_prior <- rep_len(fit$df.prior, nrow(X))
res$df_total <- fit$df.total
res$s2_prior <- rep_len(fit$s2.prior, nrow(X))
res$s2_post <- fit$s2.post
write.table(data.frame(feature = rownames(res), res, check.names = FALSE), args[2], sep = "\t", quote = FALSE,
            row.names = FALSE)
cat(sprintf("%s (%s)", as.character(packageVersion("limma")), R.version.string))
