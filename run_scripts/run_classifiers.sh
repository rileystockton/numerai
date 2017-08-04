echo 'prep_data'
python ../prep_data.py
wait

echo 'gbc_4-21 0'
python ../classifiers/gbc_4-21.py 0 &
echo 'gbc_4-21 1'
python ../classifiers/gbc_4-21.py 1 &
echo 'gbc_5-17 0'
python ../classifiers/gbc_5-17.py 0 &
echo 'gbc_5-17 1'
python ../classifiers/gbc_5-17.py 1 &
echo 'gbc_10-19 0'
python ../classifiers/gbc_10-19.py 0 &
echo 'gbc_10-19 1'
python ../classifiers/gbc_10-19.py 1 &
echo 'gbc_16-18 0'
python ../classifiers/gbc_16-18.py 0 &
echo 'gbc_16-18 1'
python ../classifiers/gbc_16-18.py 1 &
wait

echo 'logistic 0'
python ../classifiers/logistic.py 0 &
echo 'logistic 1'
python ../classifiers/logistic.py 1 &
echo 'linear_2 0'
python ../classifiers/linear_2.py 0 &
echo 'linear_2 1'
python ../classifiers/linear_2.py 1 &
wait

echo 'rfc 0 10 4'
python ../classifiers/rfc.py 0 10 4 &
echo 'rfc 1 10 4'
python ../classifiers/rfc.py 1 10 4 &
echo 'rfc 0 10 5'
python ../classifiers/rfc.py 0 10 5 &
echo 'rfc 1 10 5'
python ../classifiers/rfc.py 1 10 5 &
echo 'rfc 0 5 5'
python ../classifiers/rfc.py 0 5 5 &
echo 'rfc 1 5 5'
python ../classifiers/rfc.py 1 5 5 &
echo 'rfc 0 3 5'
python ../classifiers/rfc.py 0 3 5 &
echo 'rfc 1 3 5'
python ../classifiers/rfc.py 1 3 5 &
wait

echo 'mlp 0 3 0.001'
python ../classifiers/mlp.py 0 3 0.001 &
echo 'mlp 1 3 0.001'
python ../classifiers/mlp.py 1 3 0.001 &
echo 'mlp 0 3 0.00001'
python ../classifiers/mlp.py 0 3 0.00001 &
echo 'mlp 1 3 0.00001'
python ../classifiers/mlp.py 1 3 0.00001 &
echo 'mlp 0 5 0.001'
python ../classifiers/mlp.py 0 5 0.001 &
echo 'mlp 1 5 0.001'
python ../classifiers/mlp.py 1 5 0.001 &
echo 'mlp 0 5 0.00001'
python ../classifiers/mlp.py 0 5 0.00001 &
echo 'mlp 1 5 0.00001'
python ../classifiers/mlp.py 1 5 0.00001 &
wait

echo 'hal9000 0'
python ../ensembles/hal9000.py 0 &
echo 'hal9001 0'
python ../ensembles/hal9001.py 0 &
echo 'hal9002 0'
python ../ensembles/hal9002.py 0 &
echo 'hal9000 1'
python ../ensembles/hal9000.py 1 &
echo 'hal9001 1'
python ../ensembles/hal9001.py 1 &
echo 'hal9002 1'
python ../ensembles/hal9002.py 1 &
wait
