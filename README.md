# BYOL-CS
Development Report : https://hackmd.io/@weihsinyeh/BYOL-CS

Please click [this link](https://docs.google.com/presentation/d/1r3zxEd6IXVXq7_pQJuXwql-iKlwU1LWMqVQBsetx4rA/edit?usp=sharing) to view the slides of HW1

# Usage
To start working on this assignment, you should clone this repository into your local machine by using the following command.

    git clone https://github.com/DLCV-Fall-2024/dlcv-fall-2024-hw1-<username>.git
Note that you should replace `<username>` with your own GitHub username.

# Submission Rules
### Deadline
2024/9/27 (Fri.) 23:59

### Packages
This assignment should be done using python3.8. For a list of packages you are allowed to import in this assignment, please refer to the requirments.txt for more details.

You can run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.

If your model checkpoints are larger than GitHub’s maximum capacity (50 MB), you could download and preprocess (e.g. unzip, tar zxf, etc.) them in `hw1_download_ckpt.sh`.
* TAs will run `bash hw1_download_ckpt.sh` prior to any inference if the download script exists

# Q&A
If you have any problems related to HW1, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question to NTU COOL under HW1 Discussions

## Command Use
```shell
$ bash hw1_download_ckpt.sh
```
```shell
$ bash hw1_1.sh ./hw1_data/p1_data/office/val.csv ./hw1_data/p1_data/office/val Pb1_output.csv
```
```shell
$ bash hw1_2.sh ./hw1_data/p2_data/validation ./Pb2_evaluation
```
