#lang racket
(define (tri n)
  (define (T i)
    (cond [(= i 0) 1]
          [(= i 1) 3]
          [(even? i) (+ 1 (/ i 2.0))]
          [else (+ (T (- i 1)) (T (- i 2)) (T (+ i 1)))]))
  (for/list ([i (in-range (+ n 1))]) (T i)))
