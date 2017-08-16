my $content;

while (my $line = <STDIN>) {
  $content .= $line;
}

#for $num ($content =~ m/(.{16}'all\_.+?'.{16})/gs) {
#  print $num . "\n";
#}
#
#print "\n";

for $num ($content =~ m/\'(baseline_rec_eval.{0,10}?)\'.*?'mean_abs_err': ([0-9\-\.]+)/gs) {
  print $num . "\n";
}

for $num ($content =~ m/\'(all_.{1,10}?rec_eval.{0,10}?)\'.*?'mean_abs_err': ([0-9\-\.]+)/gs) {
  print $num . "\n";
}
