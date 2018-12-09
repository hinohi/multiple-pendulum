if ( exist("n") == 0 ) n = n_start

p dat index n u (0):(0):(sin($4)):(-cos($4)) w vec not
rep dat index n u (sin($4)):(-cos($4)):(sin($5)):(-cos($5)) w vec not

n = n + dn
print n
if ( n < n_end) reread
