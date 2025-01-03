# MaskedInnerAttentionNet: Financial Statement Line Item Substitution Network

MaskedInnerAttentionNet is a novel neural network architecture designed to handle missing data in financial statements by learning substitution patterns between line items. It leverages a unique second-order mechanism to capture relationships between financial statement entries and predict missing values based on available data.

It delivers a ~35% performance improvement against carrying over the figures from year n to year n+1 (Tested on NetIncomeLoss, as shown in the current train.py version).

## Architecture Overview

The core innovation of this architecture lies in its ability to handle missing data through an explicit masking mechanism and the use of a learnt covariance-like matrix to fill-in the gaps in the data. Eg guessing Operating Income from EBIT if the former is missing.

The general architecture is somewhat reminiscent of the Transformer's, especially in the use of multiple attention-like heads. It should be noted that the mechanism substantially differs from attention, first because it is only a 2nd order mechanism, and second because that it doesn't involve tokens.

## Installation

I will upload the requirements.txt soon.

## Usage

Just adapt the train.py file to your needs.

## Architecture Details

More to come soon on my substack (french -> https://eloidereynal.substack.com , english -> https://eligius.substack.com)

## Contributing

PRs are welcome!

## License

Feel free to do whatever you want with the code.

## Citation

If you use this architecture in your research, please cite:

```
@misc{eloisnet2024,
  author = {[Your Name]},
  title = {MaskedInnerAttentionNet: Financial Statement Line Item Substitution Network},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/edereynaldesaintmichel/masked_inner_attention}
}
```

## Acknowledgments
I used Andrej Karpathy's mini-transformer (https://github.com/karpathy/ng-video-lecture) as a template to build this net.
The data comes from [https://site.financialmodelingprep.com/] Financial Modeling Prep
