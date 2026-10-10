<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/MustafaElshani/IsoEnds/v1.0.0/docs/img/isoends_logo_dark.svg">
    <img src="https://raw.githubusercontent.com/MustafaElshani/IsoEnds/v1.0.0/docs/img/isoends_logo.svg" alt="IsoEnds: start sites, poly(A) sites and poly(A) tails" width="380">
  </picture>
</p>

# IsoEnds

**Poly(A) tail length, alternative polyadenylation and transcription start sites from
[IsoForge](https://github.com/MustafaElshani/IsoForge) long-read RNA-seq results, tested over biological replicates.**

IsoEnds reads an IsoForge run as it is. The per-read table of IsoForge holds each read's transcript, start site,
poly(A) site and tail length, so IsoEnds needs no alignments. It compares two conditions and reports, per gene:

| Analysis | What is tested |
|---|---|
| Poly(A) tails | Change in the median tail length of genes, transcripts, transcript categories and gene sets |
| Alternative polyadenylation | Tandem 3′UTR sites (PDUI), intronic sites on composite and skipped exons, internal exonic sites, 5′UTR sites, and tandem sites of non-coding genes |
| Transcription start sites | Promoter switches (DTUI), upstream promoters, alternative first exons and internal exonic starts |

Every test uses replicates as its unit, never reads. The figures on this page are IsoEnds output for a simulated
experiment of 10 units with three replicate pairs each.

```bash
isoends init --out config.yaml          # write a template, then edit it
isoends all --config config.yaml -j 4   # tails, APA, TSS and the report
```

```python
import isoends as ie

exp = ie.Experiment.from_isoforge(
    runs={"my_run": "isoforge/results"}, samples="samples.tsv", control="control", treatment="treated",
    reference_gtf="gencode.primary_assembly.annotation.gtf", outdir="isoends_out")
res = exp.run(jobs=4)
res.volcano("tandem")
```

## Contents

- [Install](#install)
- [Input](#input)
- [Poly(A) tails](#polya-tails)
- [Alternative polyadenylation](#alternative-polyadenylation)
- [Transcription start sites](#transcription-start-sites)
- [Statistics](#statistics)
- [Output](#output)
- [Python API](#python-api)
- [Command line](#command-line)
- [Development](#development)
- [Citation](#citation)

## Install

```bash
pip install isoends
```

Python 3.11 or later. `pip install "isoends[bam]"` adds pysam, for tails read from BAM files. R with limma is used
for the moderated t-test when present; without it the built-in Python implementation gives the same results.
`isoends demo` makes a synthetic experiment and analyses it.

## Input

| Input | Where | Needed for |
|---|---|---|
| IsoForge results folder | `runs: {name: {results: ...}}` | everything. IsoEnds reads `<prefix>_isoforge_read_annot.parquet` and `<prefix>_isoforge.gtf`, so run IsoForge with `--read_annot_format parquet` or `both` |
| Sample sheet | `samples:` | everything |
| Reference GTF with `Ensembl_canonical` tags (GENCODE) | `reference: {gtf: ...}` | APA and TSS |
| Characterised GTF of `isoforge-characterize` | found in the results folder or a sibling `SQANTI3` folder | categories of novel transcripts (NMD, retained intron) |
| Genome BAMs | `runs: {name: {bams: "{dataset}.bam"}}` | IsoTracks views of chosen genes |

IsoEnds uses the supported reads of the Parquet read table: reads marked `cluster_supported` whose transcript is in
the IsoForge GTF. These are the reads of IsoForge's filtered read TSV. Tail lengths come from their `polya_length`:
a read without a call has -1 and is left out, a length of 0 is a call (`tails.min_tail_nt: 1` leaves it out). A
per-read tail table or BAMs with the `pt:i` tag can be given instead (`tails:`, `tail_bams:`).

### Sample sheet

A tab- or comma-separated table with one row per dataset of the read table:

```
dataset      unit     condition  pair
A_control_1  line_A   control    1
A_treated_1  line_A   treated    1
A_control_2  line_A   control    2
A_treated_2  line_A   treated    2
B_control_1  line_B   control    1
...
```

| Column | Meaning |
|---|---|
| `dataset` | The dataset name in the IsoForge run |
| `unit` | The experiment a sample belongs to: a cell line, a donor, a time point. One unit is enough; with several, IsoEnds also tests across them |
| `condition` | The two conditions compared are named by `control` and `treatment` in the config. Datasets with any other condition are left out, so one sheet can hold several treatments |
| `pair` | Optional. Give it when each control sample has a matched treated sample; the tests within the unit are then paired. Leave the column out for independent replicates |
| `run` | Optional. The IsoForge run of the dataset, when several runs are configured and units are not named after them |

A unit needs two or more replicates of each condition to be tested on its own. To compare another pair of
conditions from the same sheet, run `isoends all --control A --treatment C --outdir out_C`.

## Poly(A) tails

Per sample, the tail of a gene or transcript is the median over its reads, and the overall tail of the sample is
the median over its transcripts with more than 20 reads. The change between conditions is tested within each unit
and across units.

```python
res.volcano("tail_genes")                 # genes; "tail_transcripts" for transcripts
res.get("tail_genes", level="within")     # the table behind it, per unit
exp.plot_tails("GENE", unit="line_A")     # the tail distribution of one gene
```

<p align="center"><img src="https://raw.githubusercontent.com/MustafaElshani/IsoEnds/v1.0.0/docs/img/example_tails.png" alt="Poly(A) tail output" width="760"></p>

(A) Per-transcript median tails of each unit by condition, with their medians. (B) Change in the median tail of
each gene, tested across units.

Also reported: the overall tail of each sample, tails by transcript category (reference type, or SQANTI3 class for
novel models), NMD and retained-intron isoforms against the coding isoform of the same gene, and the share of short
tails per gene set. Gene sets are ribosomal proteins, histones, snoRNA hosts and mitochondrial genes, or your own
(`gene_classes:`).

## Alternative polyadenylation

The 3′ ends of the reads are clustered per gene and each cluster is placed on the gene's Ensembl-canonical
transcript. Genes are Ensembl gene IDs, so two genes that share a name are never pooled.

| Class | Site | Measure |
|---|---|---|
| Tandem 3′UTR | the two most used sites that IsoForge labels `three_prime_utr` | PDUI = distal / (proximal + distal) reads |
| Composite exon | in an intron, on transcripts that read through the upstream exon | share of the gene's reads, against terminal sites |
| Skipped exon | in an intron, on a terminal exon of its own | share |
| Internal exonic | inside an exon other than the last | share |
| 5′UTR | upstream of the start codon | share |
| Intronic (all) | any intronic site | share |
| Non-coding | tandem sites of lncRNAs and novel genes | PDUI, with q over the non-coding genes |

```python
res.volcano("tandem")            # also "composite", "skipped", "internal", "five_utr", "intronic",
res.get("noncoding_tandem")      # "noncoding_tandem"
res.plot_sites("GENE", unit="line_A")
```

<p align="center"><img src="https://raw.githubusercontent.com/MustafaElshani/IsoEnds/v1.0.0/docs/img/example_apa.png" alt="Alternative polyadenylation output" width="760"></p>

Each point is a gene tested across units. Under each panel are the two isoforms compared, coloured as the points:
orange points are genes that move towards the orange isoform, teal points towards the teal one.

For each gene the tables give both sites of the tandem pair, their distance from the stop codon, the 3′UTR length
gained or lost and the transcripts ending at each site. Where two sites are equally used, the one with the lower
coordinate is taken. An intronic site is composite or skipped by the majority of the transcripts ending there, of
any gene on that strand.

| Option | Default | Alternative |
|---|---|---|
| `apa.tandem_sites` | `three_prime_utr`: sites with IsoForge's 3′UTR label | `terminal`: sites in the canonical last exon, or up to 5 kb past it |
| `apa.same_exon_check` | `false`: every pair is tested; `same_exon` and `share_distal_models_split` flag the pairs an intron splits | `true`: a pair is tested only when one annotated exon holds both sites and the transcripts ending at the distal site do not splice over the proximal one |
| `apa.across_units_same_pair` | `false`: across units, every unit's own pair | `true`: only the units whose pair matches the most common pair (`same_pair`) |
| `apa.intronic_models` | `any`: every transcript ending at the site | `gene`: only the transcripts of the site's gene |
| `apa.novel_gene_models` | `false`: genes outside the reference are not placed | `true`: placed on their best-supported IsoForge model |

<p align="center"><img src="https://raw.githubusercontent.com/MustafaElshani/IsoEnds/v1.0.0/docs/img/example_gene.png" alt="One gene" width="760"></p>

(A) Usage of the poly(A) sites of one gene (`plot_sites`). (B) Its tail distribution, with the sample medians that
are tested as ticks (`plot_tails`).

## Transcription start sites

The 5′ ends IsoForge assigns to a start site are clustered and placed on the canonical transcript in the same way.

| Class | Site | Measure |
|---|---|---|
| Promoter switch | the two most used start sites of the gene | DTUI = downstream / (upstream + downstream) reads |
| Upstream TSS | more than 1 kb upstream of the first exon | share, against the canonical first exon |
| Alternative first exon | in an intron, on transcripts with a first exon of their own | share |
| Internal exonic TSS | inside an exon other than the first | share |

```python
res.volcano("tss_switch")        # also "tss_upstream", "tss_first_exon", "tss_internal"
res.plot_sites("GENE", unit="line_A", kind="tss")
```

<p align="center"><img src="https://raw.githubusercontent.com/MustafaElshani/IsoEnds/v1.0.0/docs/img/example_tss.png" alt="Transcription start site output" width="620"></p>

Each point is a gene tested across units, with the two isoforms compared under each panel, coloured as the
points.

A 5′ end inside an internal exon can be a promoter or a truncated read, so confirm internal exonic starts with an
independent method.

## Statistics

Reads from one sample are not independent, so a test with reads as its unit calls a difference between any two
samples. IsoEnds summarises each sample first and tests over replicates.

| | Paired replicates | Unpaired replicates |
|---|---|---|
| Overall tail of a sample, within a unit | paired t-test of the sample medians | Welch t-test of the sample medians |
| Tail of a gene or transcript, within a unit | moderated one-sample t-test of the paired differences | moderated two-sample t-test of the sample medians |
| Site usage, within a unit | Cochran-Mantel-Haenszel test over the pairs | moderated two-sample t-test of each sample's usage |
| Any change, across units | moderated one-sample t-test of the per-unit changes | the same |

The moderated t-test is limma's (empirical Bayes, robust). q is the Benjamini-Hochberg-adjusted P over the genes of
a unit, or over the genes tested across units. Across units a gene must be measured in two thirds of the units.

The read-level test is still reported next to the replicate-level one, as a description only
(`q_reads_pooled` in `tails_gene_within_unit`):

<p align="center"><img src="https://raw.githubusercontent.com/MustafaElshani/IsoEnds/v1.0.0/docs/img/example_reads_vs_replicates.png" alt="Replicates and reads as units" width="760"></p>

The genes of one unit tested over its three replicate pairs (A) and with every read as a unit (B).

## Output

```
isoends_out/
  report.md          results, checks, parameters, tests, figures and table descriptions
  tables/            every table as Parquet and TSV
  figures/           PDF, SVG and PNG with captions
  provenance.json    inputs with SHA-256, versions, parameters
  logs/              counts at each step
```

| Table | Content |
|---|---|
| `tests_summary` | one row per family of tests: the test, n tested, n at q < 0.05 each way |
| `tails_gene_within_unit`, `tails_gene_across_units` | tail change per gene |
| `tails_transcript_within_unit`, `tails_transcript_across_units` | tail change per transcript, and per intron chain across units |
| `apa_tandem_*`, `apa_class_*`, `apa_ipa_*` | tandem 3′UTR, the site classes, and all intronic sites |
| `tss_switch_*`, `tss_class_*` | promoter switches and the start-site classes |
| `sites`, `tss_sites` | every poly(A) and start-site cluster with its class and reads |
| `qc_samples`, `validation` | per-sample counts and the input checks |

`report.md` describes every table.

## Python API

| | |
|---|---|
| `ie.Experiment.from_config(path)` / `.from_isoforge(runs, samples, control, treatment, reference_gtf)` | set up an analysis |
| `exp.validate()` | the input checks as a table |
| `exp.run(jobs)` / `exp.tails()` / `exp.apa()` / `exp.tss()` / `exp.report()` | run everything or one stage |
| `ie.load(outdir)` | open a finished run |
| `res.analyses()` | the analyses with the genes tested and significant |
| `res.get(analysis, level, unit)` | one analysis as a table with `effect` and `q` |
| `res.volcano(analysis)` / `res.volcano_grid(analysis)` | a volcano, or one per unit |
| `res.plot_sites(gene, unit, kind)` / `exp.plot_tails(gene, unit)` | one gene |
| `res.gene(gene)` / `res.table(name)` | every row of a gene, or any table |
| `exp.tails_of(gene, unit)` | the tail of a gene per sample |
| `exp.isotracks(gene, unit)` | the reads of a gene, drawn with IsoTracks |

Tables are pandas DataFrames and plots are Matplotlib axes. `docs/make_readme_figures.py` draws the figures of this
page, from a simulated experiment (`--simulate FOLDER`) or from a finished run of your own.

## Command line

```
isoends init      [--out isoends.yaml]      write a configuration template
isoends validate  --config config.yaml      check the inputs and the sample sheet
isoends all       --config config.yaml      tails, apa, tss and the report
isoends tails | apa | tss | report          one stage
isoends isotracks --config config.yaml --gene NAME [--unit UNIT]
isoends demo      [--out DIR] [--unpaired]  a synthetic experiment, analysed
```

`-j N` summarises N runs in parallel; summaries are cached, so later stages and reruns start from them.
`--control`, `--treatment` and `--outdir` override the config. Every parameter and its default is in the template
that `isoends init` writes.

## Development

```bash
git clone https://github.com/MustafaElshani/IsoEnds
cd IsoEnds
pip install -e ".[test]"
pytest
```

The tests check the statistics against R limma, statsmodels and SciPy, the site classes on hand-built gene models,
and the whole pipeline on synthetic experiments with known effects.

## Citation

IsoEnds is a companion tool of IsoForge. The IsoForge manuscript is in preparation. Until it is out, please cite the
IsoForge repository:

> Elshani M, Zhang Y, Harrison DJ. IsoForge, version 1.0.0. https://github.com/MustafaElshani/IsoForge

IsoEnds is released under the MIT licence.
