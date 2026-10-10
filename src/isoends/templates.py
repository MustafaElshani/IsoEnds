"""The configuration template written by `isoends init`."""

CONFIG_TEMPLATE = """\
# IsoEnds configuration. Relative paths are relative to this file. Inputs are only read.
name: my_experiment
outdir: isoends_out
# cache_dir: isoends_out/cache   # per-sample summaries; configs that share it summarise each run once

samples: samples.tsv             # columns: dataset, unit, condition and optionally pair, run
control: control                 # the two conditions compared; datasets with another condition are left out
treatment: treated

reference:
  gtf: /path/to/gencode.primary_assembly.annotation.gtf    # with Ensembl_canonical tags

runs:                            # one entry per IsoForge run
  my_run:
    results: /path/to/isoforge/results
    # bams: /path/to/alignments/{dataset}.bam               # genome BAMs, for IsoTracks views
    # tails: /path/to/tails.csv                             # a per-read tail table instead of polya_length
    # tail_columns: {read_id: read_id, tail: ePolyA_tail_length, mapq: mapping_quality}
    # tail_bams: [a.bam, b.bam]                             # or BAMs carrying the pt:i tag
    # read_annot, gtf, characterised_gtf, bed, tes_clusters, tss_clusters: single files instead of `results`

# usage_genes: [GENE1]           # report the usage (CPM) of each transcript structure of these genes
# gene_classes:                  # added to ribosomal_proteins, snoRNA_host_genes, histones and mitochondrial
#   my_set: {genes: [GENE1, GENE2]}          # or {regex: "^RPL"} or {gmt: sets.gmt, set: NAME}

figures:
  journal: cell                  # cell or nature column widths
  formats: [pdf, svg, png]
  dpi: 600
  genes: []                      # named in the volcanoes and drawn with IsoTracks

params:                          # every parameter with its default
  tails:
    require_supported: true      # only supported reads of reported transcripts (IsoForge's filtered read table)
    min_tail_nt: 0               # shortest length counted as a tail call; a read without a call has -1
    min_reads_gene: 20           # reads per sample
    min_reads_transcript: 20
    min_reads_transcript_sample: 21          # for a transcript to enter the sample's overall tail (> 20 reads)
    min_reads_isoform: 10
    min_replicates: 2            # replicate pairs, or samples per condition
    short_nt: 27
    window_nt: [30, 60]
    threshold_sets: [ribosomal_proteins]
    read_level: true             # add descriptive read-level test columns
  apa:
    cluster_nt: 25
    require_supported: true
    min_reads: 20                # per condition
    min_reads_sample: 10         # unpaired designs
    downstream_nt: 5000
    model_window_nt: 25
    tandem_sites: three_prime_utr            # IsoForge's label of the site, or terminal (the canonical last exon)
    same_exon_check: false       # true: test a pair only when one annotated exon holds both sites
    match_tolerance_nt: 50
    across_units_same_pair: false            # true: across units, only the units with the most common pair
    intronic_models: any         # models that make an intronic site composite or skipped: any or gene
    novel_gene_models: false     # true: place genes outside the reference on their best-supported model
  tss:
    enabled: true
    require_supported: true
    cluster_nt: 50
    upstream_nt: 1000
    max_upstream_nt: 50000
    model_window_nt: 50
    min_reads: 20
    min_reads_sample: 10
    match_tolerance_nt: 50
    across_units_same_pair: false
    novel_gene_models: false
    exclude_chroms: [chrM, MT]
  stats:
    alpha: 0.05
    min_units: null              # null: max(2, two thirds of the units)
    limma_backend: auto          # auto, python or r
    robust: true
    bootstrap: 10000
    seed: 1
    wilcoxon_min_units: 5
"""
