A project in which a set of socio-economic factors of high-school students in Portugal are investigated with their relation to the student's performance, potentially giving some advice to future parents who would like their children to perform well academically. Done as an assignment for the 'Introduction to Machine Learning & Data Mining' MSc course under DTU, with my classmate Mario Cesar Rodriguez.

Tl;dr: Want your child to (have a chance to statistically) perform well in school? Make sure both you and your spouse are educated (the more the better). Ensure that your child drinks the least amount of alcohol as infrequently as possible (especially on workdays). Situate yourself in an urban setting (i.e. close to school if that's the case), and provide your child with access to internet.

### Dataset
Taken from the investigation by Cortez and Silva [[1]](#1), this investigation focuses on only 13 of the 33 available attributes of the students in the dataset [[2]](#2), as only these 13 are considered by us to be within the parent's control. 

### Files
- `student-por.csv` and `student-mat.csv` contain the dataset used in this investigation, holding various attributes and grades for either Portuguese or Maths subjects respectively.  
- `1 - Feature extraction and visualisation.pdf` contains the first report for the course, which focused on feature extraction and visualisation of the dataset.
- `2 - Classification and regression.pdf` contains the final report for the course, in which classification and regression techniques are applied to the dataset.
- `report1.py` and `report2.py` contain the :snake:Python code of these reports respectively.
- `student-merge.R` contains an R script that should merge the two datasets if one would wish to, however it was found to do so erronously by Mario and I, and hence was not utilised.

### Bibliography
<a id="1">[1]</a>
P. Cortez and A. Silva, "Using data mining to predict secondary school student perofrmance," 2008. Proceedings of 5th FUture Business TEChnology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal.

<a id="2">[2]</a>
https://www.kaggle.com/datasets/uciml/student-alcohol-consumption [Accessed on 05/09/2021]
