#lang racket
(define (can_arrange arr)
  (define v (list->vector arr))
  (define n (vector-length v))
  (let loop ([i (- n 1)])
    (cond [(< i 1) -1]
          [(< (vector-ref v i) (vector-ref v (- i 1))) i]
          [else (loop (- i 1))])))
