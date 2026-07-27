#lang racket
(define (sum_squares lst)
  (for/sum ([x (in-list lst)] [i (in-naturals)])
    (cond [(= 0 (modulo i 3)) (* x x)]
          [(= 0 (modulo i 4)) (* x x x)]
          [else x])))
