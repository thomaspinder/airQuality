cwd=$(pwd)
for f in $cwd/*.ipynb; do jupyter nbconvert --to python "$f"; done
pipreqs ../../ --force
