#lang racket
(define (triples_sum_to_zero lst)
  (define v (list->vector lst))
  (define n (vector-length v))
  (for*/or ([i (in-range n)] [j (in-range (+ i 1) n)] [k (in-range (+ j 1) n)])
    (= 0 (+ (vector-ref v i) (vector-ref v j) (vector-ref v k)))))
