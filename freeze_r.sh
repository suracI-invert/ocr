echo generating new requirements.txt

pip freeze > requirements.txt

echo "-f https://download.pytorch.org/whl/torch_stable.html" >> requirements.txt