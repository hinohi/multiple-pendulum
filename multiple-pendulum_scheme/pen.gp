reset
set nokey
set xrange[-2:2]
set yrange[-2:2]
set size ratio 1

n_start=0
n_end=2**7 * 100
dn=10

if ( exists("dat") == 0 ) dat = "a.dat"

load "pen.plt"
