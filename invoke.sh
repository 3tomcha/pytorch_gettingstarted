aws lambda invoke --function-name predict --cli-binary-format raw-in-base64-out \
    --payload '{ "name": "Bob" }' \
    response.json