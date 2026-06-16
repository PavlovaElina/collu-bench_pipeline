#lang racket
(define (by_length arr)
  (define names (vector "One" "Two" "Three" "Four" "Five" "Six" "Seven" "Eight" "Nine"))
  (map (lambda (d) (vector-ref names (- d 1)))
       (reverse (sort (filter (lambda (x) (<= 1 x 9)) arr) <))))
