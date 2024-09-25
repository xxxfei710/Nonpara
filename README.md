# Nonpara
We attempted to create an interactive interface to implement the application of nonparametric methods in calculating fast radio burst energy functions and event rates.
########################################################
1. How to use?
   (1) Download all files in this repository and save them locally.
   (2) Make sure to install the modules that satisfy ``requirements.txt''.
   (3) Open the terminal in the local folder and run the command ``python app.py''.
   (4) Fill in the corresponding parameters and upload the required txt files.
   (5) Click ``Calculate''.

##########################################################

2. Page Instructions.
   (1) According to Dong et al. 2024, we set the default initial values ​​of the parameters for the 447 non-repeating FRB examples in the article.
       Of course, you can modify them according to your own requirements.
   (2) The uploaded file must be in txt format, with only two columns of data.
       The first column is the redshift value of the sample, and the second column is the energy value of the sample, just like the sample ``data.txt'' file
       Ensure that the number of data in the uploaded file corresponds to the number of samples.
   (3) We have set up a setting to allow downloading of corresponding plots, you can choose according to your needs.

##############################################################

3. Note.
   (1) The nonparametric method used here is based on Lynden-Bell 1971 and Efron & Petrosian 1992.
   (2) For additional detailed calculations, please refer to Dong et al. 2024.
   (3) Everyone is welcome to use our code, but we ask that if the provided data or software is used in any publication, the authors should cite this paper or              include the following statement in the acknowledgments:
       "The data used in our work were taken from the catalog (Dong et al., 2024), and the original data sources are cited therein."
