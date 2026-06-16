#lang racket
(define (prime_fib n)
  (define (prime? x)
    (and (> x 1)
         (for/and ([d (in-range 2 (+ 1 (integer-sqrt x)))]) (not (= 0 (modulo x d))))))
  (let loop ([a 1] [b 1] [cnt 0])
    (cond [(prime? b) (if (= (+ cnt 1) n) b (loop b (+ a b) (+ cnt 1)))]
          [else (loop b (+ a b) cnt)])))
