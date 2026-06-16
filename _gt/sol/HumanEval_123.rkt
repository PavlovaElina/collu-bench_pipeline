#lang racket
(define (get_odd_collatz n)
  (define (collatz n acc)
    (cond [(= n 1) (cons 1 acc)]
          [(even? n) (collatz (quotient n 2) acc)]
          [else (collatz (+ (* 3 n) 1) (cons n acc))]))
  (sort (remove-duplicates (collatz n '())) <))
