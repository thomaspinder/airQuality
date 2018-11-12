options(tidyverse.quiet = TRUE)
devtools::install_github('thomaspinder/usefulRFunctions', force=TRUE)
library(usefulRFunctions)
library(tidyverse, quietly=True, warn.conflicts=FALSE)

# Clear workspace
rm(list=ls())

vscode <- FALSE
if (vscode==TRUE){
  setwd("/home/pindert2/Documents/models/airQuality/DIMAQ/counties")
}

# Load MSOA data
msoa <- read.csv('src/data/cleaned/msoa.csv') %>%
	mutate(MSOA.code=as.character(MSOA.code))
air_quality <- read_csv('src/data/cleaned/aq.csv') %>%
    dplyr::filter(iso3=='GBR' & Year==2014) 
mappings <- read_csv('src/data/cleaned/msoa_to_city.csv')

colnames(mappings) <- c('msoa', 'msoa_text', 'city_id', 'city_name', 'FID')

# Join in official city list. If the join fails, city name will be extracted the the msoa text
mapped_msoa <- msoa %>%
	dplyr::left_join(mappings, by=c("MSOA.code"="msoa")) %>%
  mutate(city_name = ifelse(is.na(city_name), str_trim(str_extract(msoa_text, '\\D*')) , city_name))


write.csv(mapped_msoa, 'src/data/cleaned/mapped_msoa.csv', row.names=FALSE)