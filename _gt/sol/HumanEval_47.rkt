#lang racket
(define (median l)
  (define s (sort l <))
  (define n (length s))
  (if (odd? n)
      (list-ref s (quotient n 2))
      (/ (+ (list-ref s (- (quotient n 2) 1)) (list-ref s (quotient n 2))) 2.0)))
