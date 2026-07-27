#lang racket
(define (next_smallest lst)
  (define ds (sort (remove-duplicates lst) <))
  (if (>= (length ds) 2) (list-ref ds 1) #f))
