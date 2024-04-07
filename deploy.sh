rm -rf predict.zip
cd lambda
zip -r ../predict.zip .
cd ../
aws lambda create-function \
    --function-name predict \
    --runtime python3.10 \
    --role arn:aws:iam::092013336292:role/amplify-login-lambda-22020156 \
    --zip-file fileb://predict.zip \
    --handler predict.lambda_handler