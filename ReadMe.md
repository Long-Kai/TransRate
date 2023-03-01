## Frustratingly Easy Transferability Estimation

This is the official code of TransRate.

## Demonstration
We provide a Jupyter Notebook(*Demo.ipynb*) for the demostration of TransRate.

The calculation of TranRate requires $logdet$ on the data covariance matrix, which can be calculated using the eigenvalues of the data matrix. (see Lemma D.2).
For demonstration convenience, we extract the eigenvalues and store them in *./logs_trans*. The code for extracting the eigenvalues can be found in *./generate_transrate*.

## Citation
If you find this code useful your research, please cite our paper:
```
@inproceedings{huang2022frustratingly,
  title={Frustratingly easy transferability estimation},
  author={Huang, Long-Kai and Huang, Junzhou and Rong, Yu and Yang, Qiang and Wei, Ying},
  booktitle={International Conference on Machine Learning},
  pages={9201--9225},
  year={2022},
  organization={PMLR}
}
```
