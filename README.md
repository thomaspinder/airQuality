# Air Quality Modelling

Work done to model the effects of air quality on the health of individuals within the UK.

Currently, work is being done to creare a UK wide model of PM2.5 levels. Further down the line, data will be used from the UK BioBank to model health outcomes.

### Installation

1. Clone the repository ```git clone https://github.com/thomaspinder/airQuality.git```
2. Move into new directory ```cd airQuality```
3. Run ```make all``` to install the necessary libraries and download, clean and process the required datasets.
        * ```make all``` is comprised of three individual `make` commands, `make reqs`, `make init` and `make prep`. The former will install install the necessary python libraries onto your machine, the second will collect the three datasets before being cleaned and categorised by the final `make` command.

### Data Used

Within step 2 of the above installation instructions, two datasets will be downloaded. The first dataset comes from of the Office of National Statistics while t