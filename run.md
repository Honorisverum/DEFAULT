# run

```bash
python main.py -sig 0.005 \
               -img 32 \
               -N 5 \
               -T 10 \
               -stages 5 5 \
               -dim 100 \
               -lr 0.00005 \
               -is_train False \
               -save_every 5 \
               -save_file last.pt \
               -load_file last_weights.pt \
               -vid_dir ../
```
plus `-load_file` and `-vid_dir`, which are `None` for default


```
python main.py -img 96 \
               -dim 500 \
               -load_file last_weights.pt \
               -vid_dir ../
```