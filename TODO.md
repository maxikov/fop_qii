# TODO for code readability

* Too many magic numbers, constants whose meaning is unknown to
  reader. Consider defining them variables somewhere and give them
  meaningful names.

* Related to above: datasets are stored such that elements are looked
  up by index which makes this very dense as well. What does x[0]
  refer to ? Consider using DataFrame or Pandas or some such.

# TODO for correctness and

* Try to use existing codes from existing libraries for various tasks.

* Consider regular expressions for the parsing elements.
