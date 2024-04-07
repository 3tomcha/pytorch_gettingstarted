rm -rf predict.zip
cd lambda
zip -r ../predict.zip .
cd ../
aws lambda update-function-code \
    --function-name predict \
    --zip-file fileb://predict.zip \