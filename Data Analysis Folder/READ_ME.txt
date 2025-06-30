This folder contains the data and data processing files necessary to generate the plots present in my report. 

To view the data and perform further data processing locally, download the entire folder and open the 'Data Analysis VI.vi'. Hovering over the graph should give a description of
what they plot. Select which data sets, either Xray data or further testing, and which days to plot in the main toolbar. If selecting Xray radiation data, input an estimate
of the radiation dose in Gy/s at the target. Default is 23.7Gy/s. For an overview of what each graph shows, see the image titled 'Data Analysis VI - Annotated.png'.

Inside the Helper VIs folder are a number of SubVIs used in the main script. Without these, 'Data Analysis VI.vi' will not run (this is why the entire folder must be downloaded as one)

The folders titled 'Xray Runs' and 'Further Testing' store the experimental ic data from each measurement day in .tdms files. These can be opened in Excel by double clicking on them.
They can also be opened in Python using the nptdms library. The .tdms file format is tedious to work with so it is recommended that once a tdms file has been loaded into python, 
it should be converted into a pandas DataFrame using the .as_dataframe() function.

- Rohan
