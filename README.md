# Enhancement Strategies For Copy-Paste Generation & Localization in RGB Satellite Imagery 
<div align="center">
  
<!-- **Authors:** -->

**_¹ [Edoardo Daniele Cannas](linkedin.com/in/edoardo-daniele-cannas-9a7355146/), ² [Sriram Baireddy](https://www.linkedin.com/in/sbairedd/), ¹ [Paolo Bestagini](https://www.linkedin.com/in/paolo-bestagini-390b461b4/)_**

**_¹ [Stefano Tubaro](https://www.linkedin.com/in/stefano-tubaro-73aa9916/), ² [Edward J. Delp](https://www.linkedin.com/in/ejdelp/)_**


<!-- **Affiliations:** -->

¹ [Image and Sound Processing Laboratory](http://ispl.deib.polimi.it/), ² [Video and Image Processing Laboratory](https://engineering.purdue.edu/~ips/index.html)
</div>

This is the official code repository for the paper **Enhancement Strategies For Copy-Paste Generation & Localization in RGB Satellite Imagery**, accepted to the 2023 IEEE International Workshop on Information Forensics and Security (WIFS).  
The repository is currently **under development**, so feel free to open an issue if you encounter any problem.

<table>
  <tr>
    <td>
        <img src="assets/landsat8_no_equalization.png">
        Landsat8 sample, no equalization.
    </td>
    <td>
        <img src="assets/landsat8_uniform_equalization.png">
        Landsat8 sample, uniform equalization.
    </td>
  </tr>
  <tr>
    <td>
        <img src="assets/sentinel2a_no_equalization.png">
        Sentinel2A sample, no equalization.
    </td>
    <td>
        <img src="assets/sentinel2a_uniform_equalization.png">
        Sentinel2A sample, uniform equalization.
    </td>
  </tr>
</table>

# Getting started

## Prerequisites
In order to run our code, you need to:
1. install [conda](https://docs.conda.io/en/latest/miniconda.html)
2. create the `overhead-norm-strategies` environment using the *environment.yml* file
```bash
conda env create -f envinroment.yml
conda activate overhead-norm-strategies
```

## Data
You can download the dataset from this [link](https://www.dropbox.com/scl/fo/tr3r1ncmc0id58myc0ijf/h?rlkey=w0y5ohnya1t79smpon2w6za8m&dl=0).  
The dataset is composed of 2 folders:
1. `pristine_images`: contains the raw full resolution products (`pristine_images/full_res_products`) and the `256x256` patches extracted from them (`pristine_images/patches`);
2. `spliced_images`: contains the copy-paste images generated from the `pristine_images/patches/test_patches` using the `isplutils/create_spliced_rgb_samples.py` script.

In order to train the model, you first have to divide the dataset into training, validation and test splits.  
You can do this by running the [`notebook/Training dataset creation.ipynb`](notebooks/Training%20dataset%20creation.ipynb) notebook.
**Please notice** that these splits and patches are the ones used in the paper, but you can create your own by modifying the notebook.  

If you want to inspect the raw products, a starting point is the [Raw satellite products processing](notebooks/Raw%20satellite%20products%20processing.ipynb) notebook.  

# The whole pipeline
## Normalization strategies
All the normalization strategies used in the paper are provided as classes in the [`isplutils/data.py`](isplutils/data.py) file.  
Please notice that for the `MinPMax` strategy, we used the [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) implementation from `sklearn`.  
Statistics are learned from the training set, and then applied to the validation and test sets.  
We provide the scalers used in the paper, one for each satellite product, inside the folders of `pristine_images/full_res_products`.

## Model training
The `train_fe.py` takes care of training the models.  
You can find the network definition in the [`isplutils/network.py`](isplutils/network.py) file.  
All the hyperparameters for training are listed in the file.  
To replicate the models used in the paper, follow the [train_all.sh](bash_scripts/train_all.sh) bash script.

## Model evaluation
Inside the `data/spliced_images` folder are contained the two datasets used in the paper, i.e.:
1. `Standard Generated Dataset (SGD)`: images generated by simply normalizing the dynamics between 0 and 1 using a maximum scaling;
2. `Histogram Equalized Generated Dataset (HEGD)`: images generated by equalizing the histogram of the images using a uniform distribution.

Inside each folder, there is a Pandas DataFrame containing info on the images.  
Inside the `models` folder, we provide the models presented in the paper (both weights and definitions).  
You can replicate our results using the `test_with_AUCs.py` script. In alternative, you can run the bash script [test_all.sh](bash_scripts/test_all.sh).  
Once you have the results, use the [notebooks/Mean test results plot.ipynb](https://github.com/polimi-ispl/overhead_norm_strategies/blob/main/notebooks/Mean%20test%20results%20plot.ipynb) notebook to plot the results shown in the paper.
