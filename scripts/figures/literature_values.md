# Literature reference values (panels 2D and 5C)

Source: the spreadsheet `literature_values.ods` (saved 2026-11-04), sheets `SGN_literature`,
`IHC_literature` and `synapse_literature`. The spreadsheet is not in this repository. This file
holds every value of it that enters a figure band. DOIs checked on 2026-11-05.

## Figure bands

`_LITERATURE_REFERENCE_VALUES` in `scripts/figures/util.py` holds the literature bands of panels 2D
and 5C. Each band is (mean of the Min values, mean of the Max values) over the references in the
tables below. The code rounds the counts to integers and the synapse values to 2 decimals. All 6
bands match the .ods.

| Animal | SGNs per cochlea | IHCs per cochlea | Synapses per IHC |
|---|---|---|---|
| Mouse | 8894 to 11153 | 662 to 693 | 13.69 to 16.48 |
| Gerbil | 25173 to 26828 | 1081 to 1130 | 21.57 to 22.68 |

Min and Max are the sheet columns "Lower Bound" and "Upper Bound" ("Lower Bound Include" and
"Upper Bound Include" in `SGN_literature`). A pair is the ends of a reported range, mean -/+ SD
or SEM, or one reported value (Min = Max).

Rows in the mean:
- SGN: the rows with Include = 1. The sheet uses `AVERAGEIF(...; ">0")`.
- IHC and synapses: all rows with a Lower and an Upper Bound. The mouse "Average" and "AVG" cells
  are typed-in numbers, not formulas. They equal the mean of these rows.

Table columns:
- **Sheet entry**: the sheet row, then the "Publication" cell and the DOI or link cell as they are
  in the .ods (without non-breaking spaces). Use it if a correction in this file is wrong.
- **Publication**: the corrected first author and year. The age or method is added when one paper
  has more than one row.
- **Title**: from the Crossref record of the DOI.
- **DOI**: the checked DOI (see "DOI check").

## SGNs per cochlea

### Mouse

| Sheet entry | Publication | Title | DOI | Min | Max |
|---|---|---|---|---|---|
| row 2: Ehret, 1979; `https://doi.org/10.1002/cne.901830107` | Ehret, 1979 | Quantitative analysis of nerve fibre densities in the cochlea of the house mouse (Mus musculus) | [10.1002/cne.901830107](https://doi.org/10.1002/cne.901830107) | 9673 | 11293 |
| row 3: Webster, 1985; `10.1016/0378-5955(85)90107-8` | Webster, 1985 | The spiral ganglion and cochlear nuclei of deafness mice | [10.1016/0378-5955(85)90107-8](https://doi.org/10.1016/0378-5955(85)90107-8) | 10990 | 14164 |
| row 15: Richter et al., 2001; `10.1016/j.heares.2011.02.003` | Richter et al., 2011 (C57Bl6/129Svj, 70 d) | Unbiased counting of neurons in the cochlea of developing gerbils | [10.1016/j.heares.2011.02.003](https://doi.org/10.1016/j.heares.2011.02.003) | 10402 | 10402 |
| row 21: Carlos Thesis, 2019; `_` | Duque Afonso, 2019 (P14) | Development and Application of Tools for the Characterization of the Optogenetics Stimulation of the Cochlea | [10.53846/goediss-8106](https://doi.org/10.53846/goediss-8106) | 6916.33 | 10587.01 |
| row 22: Carlos Thesis, 2019; `_` | Duque Afonso, 2019 (3 mo) | Development and Application of Tools for the Characterization of the Optogenetics Stimulation of the Cochlea | [10.53846/goediss-8106](https://doi.org/10.53846/goediss-8106) | 6998.8 | 10641.86 |
| row 25: Keppler et al., 2021; `10.1073/pnas.2014472118` | Keppeler et al., 2021 | Multiscale photonic imaging of the native and implanted cochlea | [10.1073/pnas.2014472118](https://doi.org/10.1073/pnas.2014472118) | 8382 | 9830 |
| | **Mean (n = 6)** | | | **8893.69** | **11152.98** |

### Gerbil

| Sheet entry | Publication | Title | DOI | Min | Max |
|---|---|---|---|---|---|
| row 45: Richter et al., 2011; `https://doi.org/10.1016/j.heares.2011.02.003` | Richter et al., 2011 (P0) | Unbiased counting of neurons in the cochlea of developing gerbils | [10.1016/j.heares.2011.02.003](https://doi.org/10.1016/j.heares.2011.02.003) | 23377 | 24717 |
| row 46: Richter et al., 2011; `https://doi.org/10.1016/j.heares.2011.02.004` | Richter et al., 2011 (P7) | Unbiased counting of neurons in the cochlea of developing gerbils | [10.1016/j.heares.2011.02.003](https://doi.org/10.1016/j.heares.2011.02.003) | 24719 | 24909 |
| row 51: Cai et al., 2025; `10.1016/j.isci.2025.112929` | Cai et al., 2025 (manual) | Spiner, deep learning-based automated detection of spiral ganglion neurons in intact cochleae | [10.1016/j.isci.2025.112929](https://doi.org/10.1016/j.isci.2025.112929) | 25968 | 29327 |
| row 52: Cai et al., 2025; `10.1016/j.isci.2025.112929` | Cai et al., 2025 (Spiner) | Spiner, deep learning-based automated detection of spiral ganglion neurons in intact cochleae | [10.1016/j.isci.2025.112929](https://doi.org/10.1016/j.isci.2025.112929) | 26627 | 28357 |
| | **Mean (n = 4)** | | | **25172.75** | **26827.50** |

## IHCs per cochlea

### Mouse

| Sheet entry | Publication | Title | DOI | Min | Max |
|---|---|---|---|---|---|
| row 2: Ehret and Frankenreiter, 1977; `https://link.springer.com/article/10.1007/BF00611249` | Ehret and Frankenreiter, 1977 | Quantitative analysis of cochlear structures in the house mouse in relation to mechanisms of acoustical information processing | [10.1007/BF00611249](https://doi.org/10.1007/BF00611249) | 765 | 765 |
| row 3: Schmitz et al. 2014; `https://doi.org/10.1177/0194599813519071` | Schmitz et al., 2014 | Kanamycin-Furosemide Ototoxicity in the Mouse Cochlea: A 3-Dimensional Analysis | [10.1177/0194599813519071](https://doi.org/10.1177/0194599813519071) | 675 | 695 |
| row 9: Calos' thesis, 2019; `_` | Duque Afonso, 2019 (P14) | Development and Application of Tools for the Characterization of the Optogenetics Stimulation of the Cochlea | [10.53846/goediss-8106](https://doi.org/10.53846/goediss-8106) | 594 | 671 |
| row 10: Calos' thesis, 2019; `_` | Duque Afonso, 2019 (3 mo) | Development and Application of Tools for the Characterization of the Optogenetics Stimulation of the Cochlea | [10.53846/goediss-8106](https://doi.org/10.53846/goediss-8106) | 613 | 641 |
| | **Mean (n = 4)** | | | **661.75** | **693.00** |

### Gerbil

| Sheet entry | Publication | Title | DOI | Min | Max |
|---|---|---|---|---|---|
| row 21: Hutson et al., 2020; `doi: 10.1002/cne.24977` | Hutson et al., 2020 | Light sheet microscopy of the gerbil cochlea | [10.1002/cne.24977](https://doi.org/10.1002/cne.24977) | 1081 | 1130 |
| | **Mean (n = 1)** | | | **1081.00** | **1130.00** |

## Synapses per IHC

### Mouse

| Sheet entry | Publication | Title | DOI | Min | Max |
|---|---|---|---|---|---|
| row 3: Khimich et al, 2005; `https://www.nature.com/articles/nature03418` | Khimich et al., 2005 | Hair cell synaptic ribbons are essential for synchronous auditory signalling | [10.1038/nature03418](https://doi.org/10.1038/nature03418) | 8.9 | 10.7 |
| row 4: Brandt, Khimich, Moser, 2005; `doi: 10.1523/JNEUROSCI.3411-05.2005` | Brandt et al., 2005 (P16) | Few CaV1.3 Channels Regulate the Exocytosis of a Synaptic Vesicle at the Hair Cell Ribbon Synapse | [10.1523/JNEUROSCI.3411-05.2005](https://doi.org/10.1523/JNEUROSCI.3411-05.2005) | 13.1 | 14.3 |
| row 6: Kujawa and Liberman 2009; `doi: 10.1523/JNEUROSCI.2845-09.2009` | Kujawa and Liberman, 2009 (16 w) | Adding Insult to Injury: Cochlear Nerve Degeneration after "Temporary" Noise-Induced Hearing Loss | [10.1523/JNEUROSCI.2845-09.2009](https://doi.org/10.1523/JNEUROSCI.2845-09.2009) | 10 | 17 |
| row 7: Meyer et al., 2009; `https://doi.org/10.1038/nn.2293` | Meyer et al., 2009 | Tuning of synapse number, structure and function in the cochlea | [10.1038/nn.2293](https://doi.org/10.1038/nn.2293) | 17 | 17 |
| row 8: Libermann et al., 2011; `https://doi.org/10.1523/JNEUROSCI.3389-10.2011` | Liberman et al., 2011 | Opposing Gradients of Ribbon Size and AMPA Receptor Expression Underlie Sensitivity Differences among Cochlear-Nerve/Hair-Cell Synapses | [10.1523/JNEUROSCI.3389-10.2011](https://doi.org/10.1523/JNEUROSCI.3389-10.2011) | 16.8 | 16.8 |
| row 10: Mohammad et al., 2014; `https://doi.org/10.1038/s42003-024-06067-y` | Ismail Mohamad et al., 2024 (P4-6) | Synaptic ribbon dynamics after noise exposure in the hearing cochlea | [10.1038/s42003-024-06067-y](https://doi.org/10.1038/s42003-024-06067-y) | 28.8 | 39.8 |
| row 11: Mohammad et al., 2014; `https://doi.org/10.1038/s42003-024-06067-y` | Ismail Mohamad et al., 2024 (7-10 w) | Synaptic ribbon dynamics after noise exposure in the hearing cochlea | [10.1038/s42003-024-06067-y](https://doi.org/10.1038/s42003-024-06067-y) | 5.4 | 15.54 |
| row 12: Ohn et al., 2016; `https://doi.org/10.1073/pnas.1605737113` | Ohn et al., 2016 (P14-20) | Hair cells use active zones with different voltage dependence of Ca2+ influx to decompose sounds into complementary neural codes | [10.1073/pnas.1605737113](https://doi.org/10.1073/pnas.1605737113) | 11.19 | 12.99 |
| row 19: Guo et al., 2022; `https://doi.org/10.3389/fnmol.2022.850035` | Guo et al., 2022 (p14, see Notes) | Autophagy-Mediated Synaptic Refinement and Auditory Neural Pruning Contribute to Ribbon Synaptic Maturity in the Developing Cochlea | [10.3389/fnmol.2022.850035](https://doi.org/10.3389/fnmol.2022.850035) | 11.17 | 12.25 |
| row 20: Guo et al., 2022; `https://doi.org/10.3389/fnmol.2022.850035` | Guo et al., 2022 (P7) | Autophagy-Mediated Synaptic Refinement and Auditory Neural Pruning Contribute to Ribbon Synaptic Maturity in the Developing Cochlea | [10.3389/fnmol.2022.850035](https://doi.org/10.3389/fnmol.2022.850035) | 13.69 | 14.53 |
| row 21: Guo et al., 2022; `https://doi.org/10.3389/fnmol.2022.850035` | Guo et al., 2022 (P14) | Autophagy-Mediated Synaptic Refinement and Auditory Neural Pruning Contribute to Ribbon Synaptic Maturity in the Developing Cochlea | [10.3389/fnmol.2022.850035](https://doi.org/10.3389/fnmol.2022.850035) | 11.26 | 12.3 |
| row 22: Guo et al., 2022; `https://doi.org/10.3389/fnmol.2022.850035` | Guo et al., 2022 (P28) | Autophagy-Mediated Synaptic Refinement and Auditory Neural Pruning Contribute to Ribbon Synaptic Maturity in the Developing Cochlea | [10.3389/fnmol.2022.850035](https://doi.org/10.3389/fnmol.2022.850035) | 11.6 | 11.98 |
| row 23: Oestreicher et al., 2025; `https://doi.org/10.3389/fncel.2024.1523978` | Oestreicher et al., 2025 (3-4 w) | Noise-induced ribbon synapse loss in the mouse basal cochlear region does not reduce inner hair cell exocytosis | [10.3389/fncel.2024.1523978](https://doi.org/10.3389/fncel.2024.1523978) | 19 | 19 |
| | **Mean (n = 13)** | | | **13.69** | **16.48** |

### Gerbil

| Sheet entry | Publication | Title | DOI | Min | Max |
|---|---|---|---|---|---|
| row 27: Bovee et al., 2024; `https://doi.org/10.3390/ijms25052738` | Bovee et al., 2024 (80 and 319 d) | Cochlear Ribbon Synapses in Aged Gerbils | [10.3390/ijms25052738](https://doi.org/10.3390/ijms25052738) | 21.6 | 21.6 |
| row 29: Steenken et al., 2021; `https://doi.org/10.1016/j.neurobiolaging.2021.08.019` | Steenken et al., 2021 | Age-related decline in cochlear ribbon synapses and its relation to different metrics of auditory-nerve activity | [10.1016/j.neurobiolaging.2021.08.019](https://doi.org/10.1016/j.neurobiolaging.2021.08.019) | 20.9 | 25.6 |
| row 30: Jüchter et al, 2025; `https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0330663&utm_source=chatgpt.com#sec012` | Jüchter et al., 2025 | Speech-in-noise discriminability after noise exposure: Insights from a gerbil model of acoustic trauma | [10.1371/journal.pone.0330663](https://doi.org/10.1371/journal.pone.0330663) | 20.9 | 22.9 |
| row 36: Meyer et al., 2009; `https://doi.org/10.1038/nn.2293` | Meyer et al., 2009 | Tuning of synapse number, structure and function in the cochlea | [10.1038/nn.2293](https://doi.org/10.1038/nn.2293) | 24 | 24 |
| row 37: Zhang et al. 2018; `10.1016/j.heares.2018.03.028` | Zhang et al., 2018 | Concurrent gradients of ribbon volume and AMPA-receptor patch volume in cochlear afferent synapses on gerbil inner hair cells | [10.1016/j.heares.2018.03.028](https://doi.org/10.1016/j.heares.2018.03.028) | 22 | 22 |
| row 38: Zhang et al. 2018; `10.1016/j.heares.2018.03.028` | Zhang et al., 2018 | Concurrent gradients of ribbon volume and AMPA-receptor patch volume in cochlear afferent synapses on gerbil inner hair cells | [10.1016/j.heares.2018.03.028](https://doi.org/10.1016/j.heares.2018.03.028) | 20 | 20 |
| | **Mean (n = 6)** | | | **21.57** | **22.68** |

## DOI check

Method:
- Every DOI was resolved through the Crossref API (`api.crossref.org/works/<DOI>`). The first
  author, year and title of the record were compared with the sheet.
- The thesis DOI is a Crossref "dissertation" record of the Georg-August-Universität Göttingen. The
  repository date is 2019-08-29.
- PMC links were converted to DOIs with the NCBI ID converter.

All DOIs in the tables above resolve. Their first author and year match the Publication column.

The .ods has these errors. "In mean" marks the rows that enter a figure band.

| Sheet | Row | Sheet entry | Problem | Correction | In mean |
|---|---|---|---|---|---|
| SGN | 5 | Ferinas et al, 1994 | Spelling | Fariñas et al., 1994 | no |
| SGN | 9, 10 | Postigo, 2002 | Leading non-breaking space in the DOI | `10.1101/gad.217902` | no |
| SGN | 15 | Richter et al., 2001 | Wrong year | Richter et al., 2011. The DOI is correct. Schettino and Lauer 2013 (`10.1016/j.heares.2013.07.007`), Table 1, gives 10402 SGNs (C57Bl6/129Svj, 70 d, n = 5) for Richter et al., 2011. The paper title names only gerbils. The mouse count was not checked in the paper itself. | yes |
| SGN | 16 | `10.1016/j.heares.2013.07.007.` | Trailing period | `10.1016/j.heares.2013.07.007` | no |
| SGN | 17 | Schmitz et al., 2014 | Leading non-breaking space in the DOI | `10.1177/0194599813519071` | no |
| SGN | 18 | Johnson et al., 2011; `10.1073/pnas.1219578110` | The DOI is Johnson et al., 2013, PNAS, "Presynaptic maturation in auditory hair cells requires a critical period of sensory-independent spiking activity" | `10.1016/j.heares.2011.02.008` (Johnson et al., 2011, "TSLIM imaging and a morphometric analysis of the mouse spiral ganglion", 8626 +/- 96 SEM, n = 5) | no |
| SGN | 21-24 | Carlos Thesis, 2019; `_` | No reference | Duque Afonso, 2019, `10.53846/goediss-8106` | 21, 22 |
| SGN | 25 | Keppler et al., 2021 | Spelling | Keppeler et al., 2021 | yes |
| SGN | 26-28 | Elliot et al., 2021 | Spelling | Elliott et al., 2021 | no |
| SGN | 30, 31 | `10.1371/journal.pone.0292677`, `...0292678` | Autofill increment of row 29 | `10.1371/journal.pone.0292676` | no |
| SGN | 41-44 | `...CO;2-1` to `...CO;2-4` | Autofill increment (rows 42-44), leading non-breaking space | `10.1002/1096-9861(20000925)425:3<436::AID-CNE8>3.0.CO;2-1` | no |
| SGN | 46 | `10.1016/j.heares.2011.02.004` | Autofill increment. The DOI is Meredith et al., 2011 (crossmodal plasticity in deaf cats) | `10.1016/j.heares.2011.02.003` | yes |
| SGN | 47-49 | `10.1016/j.heares.2011.02.005` to `...007` | Autofill increment | `10.1016/j.heares.2011.02.003` | no |
| IHC | 2 | `https://link.springer.com/article/10.1007/BF00611249` | Link, not a DOI | `10.1007/BF00611249` | yes |
| IHC | 4, 5 | Tong et al., 2015; `https://www.jneurosci.org/content/35/20/7878` | Link, not a DOI | `10.1523/JNEUROSCI.2179-14.2015` | no |
| IHC | 6 | Neil et al., 2015 | Spelling | Neal et al., 2015 | no |
| IHC | 7, 8 | Stamataki et al, 2018 | Wrong year | Stamataki et al., 2006 (the DOI is correct) | no |
| IHC | 9-12 | Calos' thesis, 2019; `_` | No reference | Duque Afonso, 2019, `10.53846/goediss-8106` | 9, 10 |
| IHC | 13 | Daniel Bodmer 2021 | Not the first author | Cortada et al., 2021 (the DOI is correct) | no |
| IHC | 16 | Adams and Shulte 1996 | Spelling and year | Adams and Schulte, 1997 (the DOI is correct) | no |
| IHC | 22 | Moatti et al., 2020; PMC7687970 link | Link, not a DOI | `10.1364/BOE.402991` | no |
| IHC | 23 | Moatti et al., 2023; ScienceDirect link | Link, not a DOI | `10.1016/j.xpro.2023.102220` | no |
| Synapse | 2 | Martinez-Dunst, Michaels, Fuchs, 1997; WT mouse | The DOI is "Release Sites and Calcium Channels in Hair Cells of the Chick's Cochlea" | Species: chick. The DOI is correct | no |
| Synapse | 3 | `https://www.nature.com/articles/nature03418` | Link, not a DOI | `10.1038/nature03418` | yes |
| Synapse | 8 | Libermann et al., 2011 | Spelling | Liberman et al., 2011 | yes |
| Synapse | 9 | Kantardzhieva, LibermanF Sewell., 2013; WT mouse | The DOI is "Quantitative analysis of ribbons, vesicles, and cisterns at the cat inner hair cell synapse" | Species: cat. The DOI is correct | no |
| Synapse | 10, 11 | Mohammad et al., 2014 | Wrong first author and year | Ismail Mohamad et al., 2024 (the DOI is correct) | yes |
| Synapse | 13-15 | Yeng et al., 2020 | Spelling | Jeng et al., 2020 | no |
| Synapse | 14, 15 | `10.1113/JP280019`, `10.1113/JP280020` | Autofill increment of row 13 | `10.1113/JP280018` | no |
| Synapse | 30 | Jüchter et al, 2025; PLOS link | Link with the tracking parameter `utm_source=chatgpt.com` | `10.1371/journal.pone.0330663` | yes |
| Synapse | 31, 32 | Steenken et al., 2025; eLife reviewed-preprint link | Link, not a DOI | `10.7554/eLife.102890` (version of record 2026) | no |
| Synapse | 33 | Yin et al., 2025; GERBIL; PMC12228769 link | Link, not a DOI. The abstract describes mice, not gerbils | `10.1038/s41420-025-02607-5`. Species: mouse | no |
| Synapse | 39-41 | Gelich et al., 2016 | Spelling | Gleich et al., 2016 | no |

## Notes

- Guo et al., 2022: rows 19 and 21 both have the age P14 ("p14" and "P14"), with different values.
  Both rows are in the mouse synapse mean. Check the age of row 19 in the paper.
- Ismail Mohamad et al., 2024, row 10: P4-6 is before hearing onset (about P12 in mice). Its values
  (28.8 to 39.8) are the highest in the mouse synapse table, and they are in the mean.
