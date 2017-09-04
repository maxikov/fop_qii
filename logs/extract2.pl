my $content;

while (my $line = <STDIN>) {
  $content .= $line;
}

#for $num ($content =~ m/(.{16}'all\_.+?'.{16})/gs) {
#  print $num . "\n";
#}
#
#print "\n";

my %res = {};

@pieces = ($content =~ m/\'(baseline_rec_eval.{0,10}?)\'.*?'mean_abs_err': ([0-9\-\.]+)/gs);

while (my $thing = shift @pieces) {
  my $num = shift@pieces;
  $res{$thing} = $num;
  print $thing . "=" . $num . "\n";
}

my @pieces = ($content =~ m/\'(all_.{1,10}?rec_eval.{0,10}?)\'.*?'mean_abs_err': ([0-9\-\.]+)/gs);
while (my $thing = shift @pieces) {
  my $num = shift@pieces;
  print $thing . "=" . $num . "\n";
  $res{$thing} = $num;
}

if ($res{'all_replaced_rec_eval_test'}) {
  print sprintf("- & %0.2f & - & %0.2f & %0.2f & ", $res{'baseline_rec_eval'}, $res{'all_replaced_rec_eval'}, $res{'all_replaced_rec_eval_test'});
  print sprintf("%0.2f", $res{'all_random_rec_eval_test'} / $res{'all_replaced_rec_eval_test'}) . "\\\\\n";
}
