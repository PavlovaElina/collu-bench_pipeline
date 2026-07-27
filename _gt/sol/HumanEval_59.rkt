#lang racket
(define (largest_prime_factor n)
  (let loop ([n n] [d 2] [last 1])
    (cond [(= n 1) last]
          [(= 0 (modulo n d)) (loop (quotient n d) d d)]
          [else (loop n (+ d 1) last)])))
