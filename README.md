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
2. create the `sar-dip-anonymization` environment using the *environment.yml* file
```bash
conda env create -f envinroment.yml
conda activate overhead-norm-strategies
```

