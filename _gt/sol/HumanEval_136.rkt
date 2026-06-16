#lang racket
(define (largest_smallest_integers lst)
  (define negs (filter negative? lst))
  (define poss (filter positive? lst))
  (list (if (null? negs) #f (apply max negs))
        (if (null? poss) #f (apply min poss))))
