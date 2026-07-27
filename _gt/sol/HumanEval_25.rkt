#lang racket
(define (factorize n)
  (let loop ([n n] [d 2] [acc '()])
    (cond [(< n 2) (reverse acc)]
          [(= 0 (modulo n d)) (loop (quotient n d) d (cons d acc))]
          [else (loop n (+ d 1) acc)])))
