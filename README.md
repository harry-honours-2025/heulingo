# Large Neighborhood Prioritized Search for Combinatorial Optimization with Answer Set Programming

<!-- The heulingo solver is an ASP-based implementation of Large Neighborhood Prioritized Search (LNPS). -->
<!-- LNPS is an integration of systematic prioritized-search and SLS-based large neighborhood search -->
<!-- for solving combinatorial optimization problems. -->
<!-- LNPS starts with an initial solution and then iteratively tries to find improved solutions -->
<!-- by alternately destroying and prioritized-searching a current incumbent solution. -->
<!-- LNPS can not only find near-optimal solutions within a reasonable amount of computation time -->
<!-- but also guarantee the optimality of obtained solutions. -->

## Requirements
- [clingo](https://potassco.org/clingo/) version 5.6 or higher

## Sample sessions

### Traveling Salesperson Problem

```
python3 solver/heulingo.py --heulingo-configuration=tsp benchmark/tsp/tsp.lp benchmark/tsp/instances/dom_rand_70_300_1155482584_3.lp benchmark/tsp/configs/random_N.lp -c n=3
```

### Social Golfer Problem

```
python3 solver/heulingo.py --heulingo-configuration=sgp benchmark/sgp/golfer.lp benchmark/sgp/instances/8.lp benchmark/sgp/configs/random_w_N.lp -c n=60
```

### Sudoku Puzzle Generation

```
python3 solver/heulingo.py --heulingo-configuration=spg benchmark/spg/sudoku.lp benchmark/spg/instances/9x9.lp benchmark/spg/configs/random_N.lp -c n=14
```

### Weighted Strategic Companies

#### wstratcomp_001-050

```
python3 solver/heulingo.py --heulingo-configuration=wsc,large benchmark/wsc/wsc.lp benchmark/wsc/instances/wstratcomp_001.lp benchmark/wsc/configs/random_N.lp -c n=7
```

#### wstratcomp_051-061

```
python3 solver/heulingo.py --heulingo-configuration=wsc,medium benchmark/wsc/wsc.lp benchmark/wsc/instances/wstratcomp_051.lp benchmark/wsc/configs/random_N.lp -c n=7
```

### Shift Design

```
python3 solver/heulingo.py --heulingo-configuration=sd benchmark/sd/shift_design.lp benchmark/sd/instances/4_30m.lp benchmark/sd/configs/random_N_sign.lp -c n=20
```

