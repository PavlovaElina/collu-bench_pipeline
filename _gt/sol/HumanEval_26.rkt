#lang racket
(define (remove_duplicates lst)
  (filter (lambda (x) (= 1 (count (lambda (y) (= y x)) lst))) lst))
