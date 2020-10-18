echo "Point to my Codalab worksheet"
cl work dazcona-mrnet

echo "Upload validation paths and numpy arrays"
# cl add bundle mrnet-utils//valid
# cl add bundle mrnet-utils//valid-paths.csv
cl upload my-utils/valid/
cl upload valid-paths.csv

echo 'Remove current zip for the source code and re-create it'
rm src.zip
zip -r src.zip src

echo 'Upload source code to Codalab worksheet'
cl upload src.zip

echo 'Run predict program to generate predictions for the validation set'
cl run valid-paths.csv:valid-paths.csv valid:valid src:src "python src/predict.py -i valid-paths.csv -o predictions.csv" -n run-predictions --request-docker-image davidazcona/mrnet:1 --request-network --request-time 5m --request-memory 6g --request-gpus 1

echo 'Rename predictions generated'
cl make run-predictions/predictions.csv -n predictions-insight-v1

echo 'Verify predictions can be evaluated'
cl macro mrnet-utils/valid-eval-v1.0 predictions-insight-v1 -n predictions-insight-v1-eval --request-docker-image davidazcona/mrnet:1

echo 'REMEMBER to modify the desc of the bundle: BASELINE SYSTEM (ensemble) (Dublin City University) https://github.com/dazcona/mrnet'

echo 'Submit predictions'
cl edit predictions-insight-v1 --tags mrnet-submit