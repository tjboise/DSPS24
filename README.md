### DSPS24

Team Members: Tianjie Zhang, Alex Smith


### data

Please download the data first.

### results

please write results in the [google excel](https://docs.google.com/spreadsheets/d/1Ij8w8vqAo-dUPt97YgrExTam9bNOdL69jmSlJbwLJoI/edit#gid=0)

### some strategy

1. Train on a pre-trained model: for example, train on resnet 50 first and get the parameters. then, train again based on the parameters.
2. Log-norm transfermation: it is good for paying more attention on the small PCI value.
3. Augmentation images and PCI based on the PCI value. for example, a small PCI image would be augmented more than a high PCI image.
4. use weighted loss function to update parameters. but it seems not work.




### Links:

[DSPS24 introduction and result submission](https://dsps-1e998.web.app/)

[DSPS24 data and timeline](https://github.com/UM-Titan/DSPS24)

our work last two years: 

[DSPS22](https://github.com/tjboise/DSPS22)

[DSPS23](https://github.com/tjboise/DSPS23)


