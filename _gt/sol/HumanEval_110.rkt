#lang racket
(define (exchange lst1 lst2)
  (define odds1 (count odd? lst1))
  (define evens2 (count even? lst2))
  (if (>= evens2 odds1) "YES" "NO"))
