parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
wget 'https://opendata.arcgis.com/datasets/826dc85fb600440889480f4d9dbb1a24_3.zip?outSR=%7B%22wkid%22%3A27700%2C%22latestWkid%22%3A27700%7D' -O ../data/msoas.zip

cd ../data
unzip msoas.zip
# mv Middle*.shp msoa.shp
rm msoas.zip