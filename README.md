# Continuous and data-driven descriptor (CDDD)

Encodes a molecule as 512 continuous descriptors learned by training a network to translate between different string representations of the same structure. Winter and colleagues reasoned that a model able to convert SMILES into InChI must retain the molecule's full chemical content in the intermediate vector, giving a data-driven descriptor without any property labels. The embedding proved competitive with expert descriptors across QSAR benchmarks, and its dimensions are not individually interpretable.

This model was incorporated on 2025-11-29.Last packaged on 2025-12-24.

## Information
### Identifiers
- **Ersilia Identifier:** `eos4rw4`
- **Slug:** `cddd-onnx`

### Domain
- **Task:** `Representation`
- **Subtask:** `Featurization`
- **Biomedical Area:** `Any`
- **Target Organism:** `Any`
- **Tags:** `Embedding`, `Descriptor`, `Chemical language model`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `512`
- **Output Consistency:** `Fixed`
- **Interpretation:** 512 continuous descriptors learned by translating between molecular string representations.

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| cddd_000 | float |  | CDDD descriptor 0 of the input molecule |
| cddd_001 | float |  | CDDD descriptor 1 of the input molecule |
| cddd_002 | float |  | CDDD descriptor 2 of the input molecule |
| cddd_003 | float |  | CDDD descriptor 3 of the input molecule |
| cddd_004 | float |  | CDDD descriptor 4 of the input molecule |
| cddd_005 | float |  | CDDD descriptor 5 of the input molecule |
| cddd_006 | float |  | CDDD descriptor 6 of the input molecule |
| cddd_007 | float |  | CDDD descriptor 7 of the input molecule |
| cddd_008 | float |  | CDDD descriptor 8 of the input molecule |
| cddd_009 | float |  | CDDD descriptor 9 of the input molecule |

_10 of 512 columns are shown_
### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **DockerHub**: [https://hub.docker.com/r/ersiliaos/eos4rw4](https://hub.docker.com/r/ersiliaos/eos4rw4)
- **Docker Architecture:** `AMD64`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos4rw4.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos4rw4.zip)

### Resource Consumption
- **Model Size (Mb):** `2724`
- **Environment Size (Mb):** `958`
- **Image Size (Mb):** `5112.28`

**Computational Performance (seconds):**
- 10 inputs: `31.09`
- 100 inputs: `24.85`
- 10000 inputs: `426.01`

### References
- **Source Code**: [https://github.com/sergsb/cddd-onnx](https://github.com/sergsb/cddd-onnx)
- **Publication**: [https://doi.org/10.1039/c8sc04175j](https://doi.org/10.1039/c8sc04175j)
- **Publication Type:** `Peer reviewed`
- **Publication Year:** `2019`
- **Ersilia Contributor:** [GemmaTuron](https://github.com/GemmaTuron)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [MIT](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos4rw4
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos4rw4
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
