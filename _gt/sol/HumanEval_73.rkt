#lang racket
(define (smallest_change arr)
  (define v (list->vector arr))
  (define n (vector-length v))
  (for/sum ([i (in-range (quotient n 2))])
    (if (= (vector-ref v i) (vector-ref v (- n 1 i))) 0 1)))
