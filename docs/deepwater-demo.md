# Deep Water Demo

A Deep Water demo is available in Flow. 

1. After you have installed and built Deep Water, launch H2O. 

2. In your browser, enter the URL for the machine that is running H2O.

3. In the Model menu, select **Deep Water**.

   ![Model Menu](images/model-menu.jpg)
   
4. In the Import Files section, select the cat_dog_mouse.csv file.

   ![Cat, Dog, Mouse dataset](images/cat-dog-mouse.jpg)

5. Parse the dataset using the settings in the image below.

   ![Parse dataset](images/parse-dataset.jpg)
   
   Note that after the data is parsed, a cat\_dog\_mouse.hex file is created. 
   
   ![URI Frame Structure](images/uri-frame-structure.jpg)
 
6. Build the model.

   ![Build the Model](images/build-model.jpg)
 
   The GPU Usage Validation monitor will display while the model is being built. 
   
   ![GPU Usage Validation](images/gpu-usage-validation.jpg)
