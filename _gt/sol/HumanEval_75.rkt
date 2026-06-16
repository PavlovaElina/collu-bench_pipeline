#lang racket
(define (is_multiply_prime n)
  (define (factors n)
    (let loop ([n n] [d 2] [acc '()])
      (cond [(= n 1) (reverse acc)]
            [(= 0 (modulo n d)) (loop (quotient n d) d (cons d acc))]
            [else (loop n (+ d 1) acc)])))
  (= 3 (length (factors n))))
