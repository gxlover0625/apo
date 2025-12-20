opt_model=
opt_temp=0.7
eval_model=
eval_temp=0.3
exec_model=
exec_temp=0
template=BBH_Geo.yaml
name=BBH_Geo

python3 -u optimize.py \
    --opt-model $opt_model \
    --opt-temp $opt_temp \
    --eval-model $eval_model \
    --eval-temp $eval_temp \
    --exec-model $exec_model \
    --exec-temp $exec_temp \
    --template $template \
    --name $name