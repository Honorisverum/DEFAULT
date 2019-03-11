# run

```bash
python main.py -sig 0.01 \
               -img 64 \
               -N 5 \
               -T 10 \
               -epochs1 5 \
               -epochs2 5 \
               -dim 100 \
               -lr 0.00005 \
               -save_every 5 \
               -vid_dir .
```
plus `-load_file` and `-vid_dir`, which are `None` for default


```
python main.py -img 96 \
               -dim 500 \
               -load_file last_weights.pt \
               -vid_dir ../
```